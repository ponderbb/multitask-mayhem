import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import wandb
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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

        # make folders for the model weights and checkpoints
        self.weights_landing, self.checkpoints_landing = utils.create_model_folders(
            config_path=config,
            model_folder=self.config["model_out_path"],
            model_name=self.model_name,
            debug=self.config["debug"],
        )

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
        elif self.config["model"] == "mobilenetv3":
            raise NotImplementedError

        # loading for inference from saved weights
        if stage == "test":  # FIXME: also include validation
            if Path(self.weights_landing).exists():
                self.model.load_state_dict(torch.load(self.weights_landing + "best.pth", map_location=self.device))
            else:
                raise FileExistsError("No trained model found.")

        self.best_result = 0
        self.epoch = 0

        return super().setup(stage)

    def configure_optimizers(self) -> Any:
        """set up optimizer configurations"""
        config = self.config["optimizer"]

        if config["name"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"]
            )
        elif config["name"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"]
            )
        else:
            raise ModuleNotFoundError("Optimizer name can be [adam, sgd].")

        return super().configure_optimizers()

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if self.config["logging"]:
            wandb.log({"loss": losses})
        return losses

    def on_train_epoch_start(self) -> None:
        # initialize epoch counter
        self.epoch += 1
        if self.config["logging"]:
            wandb.log({"epoch": self.epoch})

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        images, targets = batch
        preds = self.model(images)
        results = compute_metrics(preds, targets)
        results = {key: val.item() for key, val in results.items()}
        if self.config["logging"]:
            wandb.log({"val_loss": results["map"], "best_val_loss": self.best_result})
        self.results = results["map"]

    def on_validation_end(self) -> None:
        # save model if performance is improved
        if self.epoch > 0:
            if self.best_result < self.results:
                self.best_result = self.results
                if not self.config["debug"]:
                    torch.save(self.model.state_dict(), self.weights_landing + "best.pth")
                else:
                    logging.warning("DEBUG MODE: model weights are not saved")

            logging.info("Current validation mAP: {:.4f}".format(self.results))
            logging.info("Best validation mAP: {:.4f}".format(self.best_result))

    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)
