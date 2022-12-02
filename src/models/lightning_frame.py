import logging
from functools import partial
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteHead

import src.utils as utils
from src.models.metrics import compute_metrics


class mtlMayhemModule(pl.LightningModule):
    def __init__(self, config_path, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        # load configuration scrip
        self.config = utils.load_yaml(config_path)

        # check if config file is for already trained model (with timestamp) or not
        if utils.check_if_model_timestamped(config_path):
            # load name and paths from config file
            self.model_name = str(Path(config_path).stem)
            logging.info("Model name: {}".format(self.model_name))
            self.path_dict = utils.create_paths(
                model_name=self.model_name,
                model_folder=self.config["model_out_path"],
            )
        else:
            # create timestamped name for model
            self.model_name = utils.model_timestamp(model_name=self.config["model"], attribute=self.config["attribute"])
            logging.info("Model name: {}".format(self.model_name))

            # create paths for training
            self.path_dict = utils.create_paths(
                model_name=self.model_name, model_folder=self.config["model_out_path"], assert_paths=False
            )

            # create folders for weights and checkpoints
            utils.create_model_folders(
                config_old_path=config_path,
                manifest_old_path=self.config["data_root"] + "/manifest.json",
                path_dict=self.path_dict,
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
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["detection_classes"])

        elif self.config["model"] == "fasterrcnn_mobilenetv3":

            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, weights="DEFAULT")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["detection_classes"])

        elif self.config["model"] == "ssdlite":
            self.model = ssdlite320_mobilenet_v3_large(
                pretrained=True,
                weights="DEFAULT",
            )
            in_features = det_utils.retrieve_out_channels(self.model.backbone, (320, 320))
            num_anchors = self.model.anchor_generator.num_anchors_per_location()
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)  # NOTE: stolened from a blogpost
            self.model.head.classification_head = SSDLiteClassificationHead(
                in_channels=in_features,
                num_classes=self.config["detection_classes"],
                num_anchors=num_anchors,
                norm_layer=norm_layer,
            )

        elif self.config["model"] == "maskrcnn":

            # fastercnn based on resnet50 backbone
            self.model = maskrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["segmentation_classes"])

            # maskrcnn based on resnet50 backbone
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, self.config["segmentation_classes"]
            )

        # loading for inference from saved weights
        if stage == "test":  # FIXME: also include validation
            if Path(self.path_dict["weights_path"]).exists():
                self.model.load_state_dict(torch.load(self.path_dict["weights_path"] + "/best.pth"))
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
            self.log(
                "train_loss", losses, on_step=True, on_epoch=True, logger=True, batch_size=self.config["batch_size"]
            )
        return losses

    def on_train_epoch_start(self) -> None:
        # initialize epoch counter
        self.epoch += 1
        if self.config["logging"]:
            self.log(
                "epoch_sanity",
                torch.as_tensor(self.epoch, dtype=torch.float32),
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            # wandb.log({"epoch": self.epoch})

    def on_validation_start(self) -> None:
        self.val_targets = []
        self.val_preds = []

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch
        preds = self.model(images)
        # self.log_validation_images(prediction=preds, target=targets)
        targets = list(targets)
        self.val_targets.extend(targets)
        self.val_preds.extend(preds)

    def on_validation_epoch_end(self) -> None:
        # skip sanity check
        if self.epoch > 0:
            # compute metrics
            results = compute_metrics(preds=self.val_preds, targets=self.val_targets)

            # extract mAP overall and for each class
            results_map = results["map"].item()
            classes_map = results["map_per_class"].tolist()

            results_classes_map = {self.class_lookup["bbox_rev"][idx + 1]: map for idx, map in enumerate(classes_map)}

            if self.config["logging"]:
                self.log("val_map", results_map, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log("class_map", results_classes_map, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # save model if performance is improved
            if self.best_result < results["map"]:
                self.best_result = results["map"]
                if not self.config["debug"]:
                    logging.info("Saving model weights.")
                    torch.save(self.model.state_dict(), self.path_dict["weights_path"] + "/best.pth")
                else:
                    logging.warning("DEBUG MODE: model weights are not saved")

            logging.info("Current validation mAP: {:.6f}".format(results["map"]))
            logging.info("Best validation mAP: {:.6f}".format(self.best_result))

    def log_validation_images(self, prediction, target):
        image = T.ToPILImage()(prediction[0].mul(255).type(torch.uint8))
        scores = prediction[0]["scores"]
        score_mask = scores > 0.8

        # boxes_filtered = prediction[0]["boxes"][score_mask]
        masks_filtered = prediction[0]["masks"][score_mask]
        labels_filtered = prediction[0]["labels"][score_mask]

        img = wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": masks_filtered,
                    "class_labels": self.class_lookup["sseg_rev"],
                },
                "ground_truth": {
                    "mask_data": target[0]["masks"],
                    "class_labels": self.class_lookup["sseg_rev"],
                },
            },
        )

        self.log("Val_Image", img, on_step=False, on_epoch=True, logger=True)
