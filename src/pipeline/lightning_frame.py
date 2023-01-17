import logging
import os
from typing import Any

import pytorch_lightning as pl
import torch
import wandb
from torchmetrics.functional.classification import binary_jaccard_index

import src.utils as utils
from src.features.metrics import compute_metrics
from src.models.model_loader import ModelLoader
from src.pipeline.lightning_utils import plUtils


class mtlMayhemModule(pl.LightningModule):
    def __init__(self, config_path, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        # load configuration scrip and classes
        self.config = utils.load_yaml(config_path)
        self.class_lookup = utils.load_yaml("configs/class_lookup.yaml")

        # initialize tasks, metrics and losses
        self.model_tasks, self.val_metric, self.loss = ModelLoader.get_type(self.config)

        self.model_name, self.path_dict = plUtils.resolve_paths(config_path)

        logging.info(
            "Running model instance with ID: {}\ntask(s): {}\nmetric(s): {}".format(
                self.model_name, self.model_tasks, self.val_metric
            )
        )

    def setup(self, stage: str) -> None:
        # load model
        self.model = ModelLoader.grab_model(config=self.config)

        # initialize variables
        self.best_result = 0
        self.epoch = 0

        # turn of logging plots for testing or validation
        if stage == "test" or stage == "validate":
            self.config["logging"] = False

            if os.listdir(self.path_dict["weights_path"]):
                self.model.load_state_dict(torch.load(self.path_dict["weights_path"] + "/best.pth"))
            else:
                raise FileExistsError("No trained model found.")

        elif stage == "fit":
            if os.path.exists(self.path_dict["weights_path"]):
                if os.listdir(self.path_dict["weights_path"]):
                    logging.info("Resuming training from checkpoint")
                    raise NotImplementedError("Resume training from checkpoint not implemented yet.")
            else:
                logging.info("No trained model found, strap in for the ride.")

        # update configuration of hyperparams in wandb
        if self.config["logging"]:
            wandb.config["model_type"] = self.model_tasks
            wandb.config.update(self.config)

    def configure_optimizers(self) -> Any:
        # configurations for optimizer and scheduler
        optim_config = self.config["optimizer"]
        lr_config = self.config["lr_scheduler"]

        # loss balancing params
        self._initialize_loss_balancing()

        # choose optimizer
        if optim_config["name"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=optim_config["lr"],
                momentum=optim_config["momentum"],
                weight_decay=optim_config["weight_decay"],
            )
        elif optim_config["name"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=optim_config["lr"],
                weight_decay=optim_config["weight_decay"],
            )
        else:
            raise ModuleNotFoundError("Optimizer name can be [adam, sgd].")

        # choose scheduler
        if lr_config["name"] == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_config["step_size"],
                gamma=lr_config["gamma"],
            )
        elif lr_config["name"] is None:
            self.lr_scheduler = None
        else:
            raise ModuleNotFoundError("Learning rate scheduler name can be [steplr or None].")

        return [self.optimizer], [self.lr_scheduler]

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        """forward pass and loss calculation
        images: [B,C,H,W]
        targets: dictionary with keys "boxes", "masks", "labels"
        """
        images, targets = batch
        self.train_loss = {}

        # format to [B,C,H,W] tensor
        if isinstance(images, tuple):
            images = plUtils.tuple_of_tensors_to_tensor(images)
            # targets only contain masks as torch.BoolTensor
            target_masks = tuple([target["masks"] for target in targets])
            target_masks = plUtils.tuple_of_tensors_to_tensor(target_masks)

        # model specific forward pass #
        if len(self.model_tasks) == 1:
            if "detection" in self.model_tasks:
                preds = self.model(images, targets)
                self.train_loss["det"] = sum(loss for loss in preds.values())
                self.train_loss["master"] = self.train_loss["det"]

            if "segmentation" in self.model_tasks:
                preds = self.model(images)
                self.train_loss["seg"] = self.loss["segmentation"](preds["out"], target_masks.type(torch.float32))
                self.train_loss["master"] = self.train_loss["seg"]
        else:
            preds = self.model(images, targets)

            self.train_loss["det"] = sum(loss for loss in preds["detection"].values()) / len(
                preds["detection"].values()
            )
            self.train_loss["seg"] = self.loss["segmentation"](preds["segmentation"], target_masks.type(torch.float32))

            if self.config["logging"]:
                self.log(
                    "train_loss_wo",
                    self.train_loss,
                    on_step=True,
                    on_epoch=True,
                    batch_size=self.config["batch_size"],
                )

            self._loss_balancing_step()

        # _endof model specific forward pass #

        if self.config["logging"]:
            self.log(
                "train_loss",
                self.train_loss,
                on_step=True,
                on_epoch=True,
                batch_size=self.config["batch_size"],
            )

        return self.train_loss["master"]

    def on_train_epoch_end(self) -> None:
        self._loss_balancing_epoch_end()

    def on_train_epoch_start(self) -> None:
        # initialize epoch counter
        self.epoch += 1
        if self.config["logging"]:
            self.log(
                "epoch_sanity",
                torch.as_tensor(self.epoch, dtype=torch.float32),
                on_epoch=True,
            )

        if self.config["weight"] == "dynamic":
            self._calculate_lambda_weight()

    def on_validation_start(self) -> None:
        self.val_images = []
        self.val_targets = []
        self.val_target_masks = []
        self.val_losses = {}
        self.val_preds = {"det": [], "seg": []}
        self.metric_box = self.loss["detection"]

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        """forward pass and loss calculation
        images: [B,C,H,W]
        targets: dictionary with keys "boxes", "masks", "labels"
        """
        images, targets = batch

        # format to [B,C,H,W] tensor
        if isinstance(images, tuple):
            images = plUtils.tuple_of_tensors_to_tensor(images)
            # targets only contain masks as torch.BoolTensor
            target_masks = tuple([target["masks"] for target in targets])
            target_masks = plUtils.tuple_of_tensors_to_tensor(target_masks)

        preds = self.model(images)

        self.val_images.extend(images)
        self.val_targets.extend(list(targets))
        self.val_target_masks.extend(target_masks.long())

        if "detection" in self.val_metric.keys():
            if len(self.model_tasks) == 1:
                self.val_preds["det"].extend(preds)
            else:
                self.val_preds["det"].extend(preds["detection"])

        if "segmentation" in self.val_metric.keys():
            if len(self.model_tasks) == 1:
                self.val_preds["seg"].extend(preds["out"])
            else:
                self.val_preds["seg"].extend(preds["segmentation"])

    def on_validation_epoch_end(self) -> None:
        # skip sanity check
        if self.epoch > 0:

            val_loss = {}

            if "detection" in self.model_tasks:
                # compute metrics
                results = compute_metrics(
                    metric_box=self.metric_box, preds=self.val_preds["det"], targets=self.val_targets
                )

                # extract mAP overall and for each class
                if self.config["class_metrics"]:
                    classes_map = results["map_per_class"].tolist()
                    val_loss["det_class"] = {
                        self.class_lookup["bbox_rev"][idx + 1]: map for idx, map in enumerate(classes_map)
                    }

                val_loss["det"] = results["map"].item()
                self.current_result = val_loss["det"]

            if "segmentation" in self.model_tasks:
                segmentation_losses = []
                for (pred, target) in zip(self.val_preds["seg"], self.val_target_masks):
                    seg_loss = binary_jaccard_index(
                        preds=pred,
                        target=target,
                        ignore_index=self.class_lookup["sseg"]["background"],
                    )
                    segmentation_losses.append(seg_loss)
                segmentation_losses = torch.nan_to_num(torch.stack(segmentation_losses))  # FIXME: zeroed out NaNs
                val_loss["seg"] = torch.mean(segmentation_losses)
                self.current_result = val_loss["seg"]

            if len(self.model_tasks) != 1:
                val_loss["master"] = val_loss["det"] * 0.5 + val_loss["seg"] * 0.5
                self.current_result = val_loss["master"]

            if self.config["logging"]:
                self.log(
                    "val_loss",
                    val_loss,
                    on_epoch=True,
                    batch_size=self.config["batch_size"],
                )
                self.log(
                    "earlystop",
                    self.current_result,
                    on_epoch=True,
                    batch_size=self.config["batch_size"],
                )  # BUG: redundant callback bugfix

            self._save_model(save_on="max")

            if self.config["logging"]:
                plUtils._log_validation_images(
                    epoch=self.epoch,
                    class_lookup=self.class_lookup,
                    model_type=self.model_tasks,
                    sanity_epoch=self.config["sanity_epoch"],
                    sanity_num=self.config["sanity_num"],
                    image_batch=self.val_images,
                    prediction_batch=self.val_preds,
                    target_batch=self.val_targets,
                )

            logging.info("Current validation: {:.6f}".format(self.current_result))
            logging.info("Best validation: {:.6f}".format(self.best_result))

    def _save_model(self, save_on: str):
        """save model on best validation result
        DEBUG_MODE: model is not saved!
        """

        save_model = False
        if save_on == "max":
            save_model = self.best_result < self.current_result
        if save_on == "min":
            save_model = self.best_result > self.current_result

        if save_model:
            self.best_result = self.current_result
            self.log("best_val", self.best_result)

            if not self.config["debug"]:
                logging.info("Saving model weights.")
                torch.save(
                    self.model.state_dict(),
                    self.path_dict["weights_path"] + "/best.pth",
                )
            else:
                logging.warning("DEBUG MODE: model weights are not saved")

    def _initialize_loss_balancing(self):
        task_count = len(self.model_tasks)
        if task_count > 1:
            if self.config["weight"] == "uncertainty":
                weights_init_tensor = torch.Tensor([-0.7] * task_count)
                self.logsigma = torch.nn.parameter.Parameter(weights_init_tensor, requires_grad=True)

            if self.config["weight"] in ["equal", "dynamic"]:
                self.temperature = 2.0
                self.lambda_weight = torch.ones(task_count)
        else:
            logging.info("Single task detected, no loss weighting applied.")

        return None

    def _calculate_lambda_weight(self):
        if self.epoch > 2:
            w = []
            for n1, n2 in zip(self.nMinus1_loss, self.nMinus2_loss):
                w.append(n1 / n2)
            w = torch.softmax(torch.tensor(w) / self.temperature, dim=0)
            self.lambda_weight = len(self.model_tasks) * w.numpy()

    def _loss_balancing_step(self):

        task_count = len(self.model_tasks)

        if task_count > 1:
            train_loss_list = list(self.train_loss.values())

            if self.config["weight"] == "uncertainty":
                balanced_loss = [
                    1 / (2 * torch.exp(w)) * train_loss_list[i] + w / 2 for i, w in enumerate(self.logsigma)
                ]

                if self.config["logging"]:
                    self.log(
                        "logsigma",
                        {"det": self.logsigma[0], "seg": self.logsigma[1]},
                        on_epoch=True,
                    )

            if self.config["weight"] in ["equal", "dynamic"]:
                balanced_loss = [w * train_loss_list[i] for i, w in enumerate(self.lambda_weight)]

                if self.config["logging"]:
                    self.log(
                        "lambda_weight",
                        {"det": self.lambda_weight[0], "seg": self.lambda_weight[1]},
                        on_epoch=True,
                    )

            self.train_loss = {"master": sum(balanced_loss), "det": balanced_loss[0], "seg": balanced_loss[1]}

    def _loss_balancing_epoch_end(self):
        if self.config["weight"] == "dynamic":

            train_loss_detached = {k: v.detach() for k, v in self.train_loss.items()}
            train_loss_detached.pop("master")

            if self.epoch == 1:
                self.nMinus2_loss = list(train_loss_detached.values())
            elif self.epoch == 2:
                self.nMinus1_loss = list(train_loss_detached.values())
            else:
                self.nMinus2_loss = self.nMinus1_loss
                self.nMinus1_loss = list(train_loss_detached.values())
