import logging
import os
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torchmetrics.functional.classification import binary_jaccard_index

import src.utils as utils
from src.features.gradient_framework import (
    cagrad,
    grad2vec,
    graddrop,
    overwrite_grad,
    pcgrad,
)
from src.features.loss_balancing import LossBalancing
from src.features.metrics import compute_metrics
from src.models.model_loader import ModelLoader
from src.pipeline.lightning_utils import plUtils


class mtlMayhemModule(pl.LightningModule):
    def __init__(self, config_path, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        # load configuration scrip and classes
        self.config = utils.load_yaml(config_path)
        self.class_lookup = utils.load_yaml("configs/class_lookup.yaml")

        if self.config["grad_method"] is not None:
            self.automatic_optimization = False
            self.customOptimizer = None

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
        elif lr_config["name"] == "onplateau":
            # assert not any([param == None for param in lr_config.values()]), "Missing params for learning rate scheduler."
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=lr_config["gamma"],
                patience=lr_config["patience"],
                cooldown=lr_config["cooldown"],
                verbose=True,
            )

        else:
            raise ModuleNotFoundError("Learning rate scheduler not found.")

        # loss balancing params
        self.balancer = LossBalancing(
            config=self.config,
            model_tasks=self.model_tasks,
            model=self.model,
            device=self.device,
            lr_scheduler=self.lr_scheduler,
            optimizer=self.optimizer,
            meta_dataloader=self.trainer.datamodule.meta_dataloader(),
            logger=self.log,
        )

        # apply gradient methods
        if self.config["grad_method"] is not None:
            self.rng = np.random.default_rng()
            self.grad_dims = []
            for mm in self.model.shared_modules():
                for param in mm.parameters():
                    self.grad_dims.append(param.data.numel())
            self.grads = torch.Tensor(sum(self.grad_dims), len(self.model_tasks)).to(self.device)

        return [self.optimizer], [self.lr_scheduler]

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
            self.balancer.calculate_lambda_weight(epoch=self.epoch)

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
            # targets only contain masks as torch.BoolTensor, needed conversion for single deeplabv3
            target_masks = plUtils.tuple_of_tensors_to_tensor(tuple([target["masks"] for target in targets]))

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

            self.balancer.update_meta_weights(train_image=images, train_target=targets)

            preds = self.model(images, targets)

            self.train_loss["det"] = sum(loss for loss in preds["detection"].values()) / len(
                preds["detection"].values()
            )
            self.train_loss["seg"] = preds["segmentation"]

            if self.config["logging"]:
                self.log(
                    "train_loss_wo",
                    self.train_loss,
                    on_step=True,
                    on_epoch=True,
                    batch_size=self.config["batch_size"],
                )

            self.train_loss = self.balancer.loss_balancing_step(train_loss=self.train_loss)

            self.train_loss_tmp = [self.train_loss["det"], self.train_loss["seg"]]

        # _endof model specific forward pass #

        if self.config["logging"]:
            self.log(
                "train_loss",
                self.train_loss,
                on_step=True,
                on_epoch=True,
                batch_size=self.config["batch_size"],
            )

        if self.config["grad_method"] is not None:

            """Manual gradient calculation with gradient clipping

            Within self.gradient_step():
            - using self.manual_backward(loss)
            - different gradient calculation for [graddrop, pcgrad, cagrad]
            - overwrite model gradients

            """

            self.optimizer.zero_grad()
            self.gradient_step()
            self.clip_gradients(self.optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            self.optimizer.step()

        else:

            return self.train_loss["master"]

    def on_train_epoch_end(self) -> None:
        self.balancer.loss_balancing_epoch_end(train_loss=self.train_loss, epoch=self.epoch)

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

        # validation pass if no target passed, no backpropagation
        preds = self.model(images)

        # collect validation relevant objects
        self.val_images.extend(images)
        self.val_targets.extend(list(targets))
        self.val_target_masks.extend(target_masks.long())

        # collect validation outputs (different for MTL and single task)
        if "detection" in self.val_metric.keys():
            if len(self.model_tasks) == 1:
                self.val_preds["det"].extend(preds)
            else:
                self.val_preds["det"].extend(preds["detection"])

        # collect validation outputs (different for MTL and single task)
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

            self.best_result = plUtils.save_model(
                best_result=self.best_result,
                current_result=self.current_result,
                model=self.model,
                debug=self.config["debug"],
                path_dict=self.path_dict,
                save_on="max",
                logger=self.log,
            )

            if self.config["logging"]:
                plUtils.log_validation_images(
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

    def gradient_step(self):
        for i in range(len(self.model_tasks)):
            self.manual_backward(loss=self.train_loss_tmp[i], retain_graph=True)
            grad2vec(self.model, self.grads, self.grad_dims, i)
            self.model.zero_grad_shared_modules()

        if self.config["grad_method"] == "graddrop":
            g = graddrop(self.grads)
        elif self.config["grad_method"] == "pcgrad":
            g = pcgrad(self.grads, self.rng, len(self.model_tasks))
        elif self.config["grad_method"] == "cagrad":
            g = cagrad(self.grads, len(self.model_tasks), 0.4, rescale=1)

        overwrite_grad(self.model, g, self.grad_dims, len(self.model_tasks))
