import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import wandb
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import src.utils as utils
from src.models.metrics import compute_metrics


class mtlMayhemModule(pl.LightningModule):
    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        # load configuration scrip
        self.config = utils.load_yaml(config)

        # create unqiue timestamped name with optinal attributes
        self.model_name = utils.model_timestamp(model_name=self.config["model"], attribute=self.config["attribute"])

        # make folders for the model weights and checkpoints, move manifest and config
        (self.weights_landing, self.checkpoints_landing, self.model_landing,) = utils.create_model_folders(
            config_path=config,
            manifest_path=self.config["data_root"] + "/manifest.json",
            model_folder=self.config["model_out_path"],
            model_name=self.model_name,
            debug=self.config["debug"],
        )

        self.class_lookup = utils.load_yaml("configs/class_lookup.yaml")

    def setup(self, stage: str) -> None:
        # turn of logging plots for testing or validation
        if stage == "test" or stage == "validate":
            self.config["logging"] = False

        # update configuration of hyperparams in wandb
        if self.config["logging"]:
            wandb.config.update(self.config)

        if self.config["model"] == "fasterrcnn":
            self.model = fasterrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["num_classes"])
        elif self.config["model"] == "fasterrcnn_v2":
            self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True, weights="DEFAULT")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["num_classes"])
        elif self.config["model"] == "mobilenetv3":
            raise NotImplementedError
        elif self.config["model"] == "maskrcnn":
            self.model = maskrcnn_resnet50_fpn(predtrained=True, weights="DEFAULT")

        # loading for inference from saved weights
        if stage == "test":  # FIXME: also include validation
            if Path(self.weights_landing).exists():
                self.model.load_state_dict(torch.load(self.weights_landing + "/best.pth"))
            else:
                raise FileExistsError("No trained model found.")

        self.best_result = 0
        self.epoch = 0

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
                momentum=optim_config["momentum"],
                weight_decay=optim_config["weight_decay"],
            )
        else:
            raise ModuleNotFoundError("Optimizer name can be [adam, sgd].")

        # choose scheduler
        if lr_config["name"] == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=lr_config["step_size"], gamma=lr_config["gamma"]
            )
        elif lr_config["name"] is None:
            self.lr_scheduler = None
        else:
            raise ModuleNotFoundError("Learning rate scheduler name can be [steplr or None].")

        return [self.optimizer], [self.lr_scheduler]

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if self.config["logging"]:
            self.log("train_loss", losses, on_step=True, on_epoch=True, logger=True)
        return losses

    def on_train_epoch_start(self) -> None:
        # initialize epoch counter
        self.epoch += 1
        if self.config["logging"]:
            self.log("epoch_sanity", self.epoch, on_epoch=True)
            # wandb.log({"epoch": self.epoch})

    def on_validation_start(self) -> None:
        self.val_targets = []
        self.val_preds = []

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch
        targets = list(targets)
        preds = self.model(images)
        self.val_targets.extend(targets)
        self.val_preds.extend(preds)

    def on_validation_epoch_end(self) -> None:
        # skip sanity check
        if self.epoch > 0:
            # compute metrics
            results = compute_metrics(preds=self.val_preds, targets=self.val_targets)
            results = {key: val.item() for key, val in results.items()}
            if self.config["logging"]:
                self.log("val_result", results["map"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # save model if performance is improved
            if self.best_result < results["map"]:
                self.best_result = results["map"]
                if not self.config["debug"]:
                    logging.info("Saving model weights.")
                    torch.save(self.model.state_dict(), self.weights_landing + "/best.pth")
                else:
                    logging.warning("DEBUG MODE: model weights are not saved")

            logging.info("Current validation mAP: {:.6f}".format(results["map"]))
            logging.info("Best validation mAP: {:.6f}".format(self.best_result))
