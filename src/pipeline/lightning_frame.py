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
        self.model_type, self.val_metric, self.loss = ModelLoader.get_type(self.config)

        self.model_name, self.path_dict = plUtils.resolve_paths(config_path)

        logging.info(
            "Running model instance with ID: {}\ntask(s): {}\nmetric(s): {}".format(
                self.model_name, self.model_type, self.val_metric
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
            wandb.config["model_type"] = self.model_type
            wandb.config.update(self.config)

    def configure_optimizers(self) -> Any:
        # configurations for optimizer and scheduler
        optim_config = self.config["optimizer"]
        lr_config = self.config["lr_scheduler"]

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
        train_loss = {}

        # format to [B,C,H,W] tensor
        if isinstance(images, tuple):
            images = plUtils.tuple_of_tensors_to_tensor(images)
            # targets only contain masks as torch.BoolTensor
            target_masks = tuple([target["masks"] for target in targets])
            target_masks = plUtils.tuple_of_tensors_to_tensor(target_masks)

        # model specific forward pass #
        if len(self.model_type) == 1:
            if "detection" in self.model_type:
                preds = self.model(images, targets)
                train_loss["det"] = sum(loss for loss in preds.values())
                train_loss["master"] = train_loss["det"]

            if "segmentation" in self.model_type:
                preds = self.model(images, targets)
                train_loss["seg"] = self.loss["segmentation"](preds["out"], target_masks.type(torch.float32))
                train_loss["master"] = train_loss["seg"]
        else:
            preds = self.model(images, targets)

            train_loss["det"] = sum(loss for loss in preds["detection"].values()) / len(preds["detection"].values())
            train_loss["seg"] = self.loss["segmentation"](preds["segmentation"], target_masks.type(torch.float32))

            train_loss["master"] = train_loss["det"] * 0.5 + train_loss["seg"] * 0.5

        # _endof model specific forward pass #

        if self.config["logging"]:
            for key, value in train_loss.items():
                self.log(
                    "train_{}".format(key),
                    value,
                    on_step=True,
                    on_epoch=True,
                    batch_size=self.config["batch_size"],
                )

        return train_loss["master"]

    def on_train_epoch_start(self) -> None:
        # initialize epoch counter
        self.epoch += 1
        if self.config["logging"]:
            self.log(
                "epoch_sanity",
                torch.as_tensor(self.epoch, dtype=torch.float32),
                on_epoch=True,
            )

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
        if "detection" in preds.keys():
            self.val_preds["det"].extend(preds["detection"])
        if "segmentation" in preds.keys():
            self.val_preds["seg"].extend(preds["segmentation"])

    def on_validation_epoch_end(self) -> None:
        # skip sanity check
        if self.epoch > 0:

            val_loss = {}

            if "detection" in self.model_type:
                # compute metrics
                results = compute_metrics(
                    metric_box=self.metric_box, preds=self.val_preds["det"], targets=self.val_targets
                )

                # extract mAP overall and for each class
                val_loss["det"] = results["map"].item()

                classes_map = results["map_per_class"].tolist()
                val_loss["det_class"] = {
                    self.class_lookup["bbox_rev"][idx + 1]: map for idx, map in enumerate(classes_map)
                }
                val_loss["master"] = val_loss["det"]

            if "segmentation" in self.model_type:
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
                val_loss["master"] = val_loss["seg"]

            if len(self.model_type) != 1:
                val_loss["master"] = val_loss["det"] * 0.5 + val_loss["seg"] * 0.5

            if self.config["logging"]:
                for key, value in val_loss.items():
                    self.log(
                        "val_{}".format(key),
                        value,
                        on_epoch=True,
                        batch_size=self.config["batch_size"],
                    )

            self.current_result = val_loss["master"]

            self._save_model(save_on="max")

            if self.config["logging"]:
                plUtils._log_validation_images(
                    epoch=self.epoch,
                    class_lookup=self.class_lookup,
                    model_type=self.model_type,
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
