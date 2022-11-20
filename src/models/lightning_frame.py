import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import wandb
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor)
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import src.utils as utils
from src.models.metrics import compute_metrics

# from torchvision.models.detection.roi_heads import fastrcnn_loss


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

        # # update configuration of hyperparams in wandb
        # if self.config["logging"]:
        #     wandb.init(dir=self.model_landing, config=self.config)

        if self.config["model"] == "fasterrcnn":
            self.model = fasterrcnn_resnet50_fpn(pretrained=True, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["num_classes"])
        elif self.config["model"] == "mobilenetv3":
            raise NotImplementedError

        # loading for inference from saved weights
        if stage == "test":  # FIXME: also include validation
            if Path(self.weights_landing).exists():
                self.model.load_state_dict(torch.load(self.weights_landing + "best.pth"))
            else:
                raise FileExistsError("No trained model found.")

        self.best_result = 0
        self.epoch = 0

    def configure_optimizers(self) -> Any:
        """set up optimizer configurations"""
        config = self.config["optimizer"]

        if config["name"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )
        elif config["name"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )
        else:
            raise ModuleNotFoundError("Optimizer name can be [adam, sgd].")

        return self.optimizer

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if self.config["logging"]:
            wandb.log(
                {
                    "train_loss": losses,
                    "epoch": self.epoch,
                    "batch": batch_idx,
                }
            )
        return losses

    def on_train_epoch_start(self) -> None:
        # initialize epoch counter
        self.epoch += 1
        if self.config["logging"]:
            wandb.log({"epoch": self.epoch})

        # wandb.define_metric(name="epoch", step_metric="epoch")  # can be changed to batch

    def on_validation_start(self) -> None:
        self.val_targets = []
        self.val_preds = []

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch
        targets = list(targets)
        preds = self.model(images)
        self.val_targets.extend(targets)
        self.val_preds.extend(preds)

    def on_validation_end(self) -> None:
        # skip sanity check
        if self.epoch > 0:
            # compute metrics
            results = compute_metrics(preds=self.val_preds, targets=self.val_targets)
            results = {key: val.item() for key, val in results.items()}
            if self.config["logging"]:
                wandb.log(
                    {
                        "val_map": results["map"],
                        "val_all": results,
                        "epoch": self.epoch,
                    }
                )

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

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch

        boxes = targets[0]["boxes"]
        labels = targets[0]["labels"]
        masks = targets[0]["masks"]
        label_names = [self.class_lookup["bbox_rev"][label.item()] for label in labels]

        img = images[0].mul(255).type(torch.uint8)
        drawn_image = draw_bounding_boxes(img, boxes, label_names)
        drawn_image = draw_segmentation_masks(drawn_image, masks, alpha=0.5, colors="green")
        image_pil = T.ToPILImage()(drawn_image)
        image_pil.save(self.model_landing + "inf/{}.png".format(batch_idx))

        return super().test_step(*args, **kwargs)
