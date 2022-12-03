import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import wandb
from torch.autograd import Variable
from torchmetrics.functional import dice, jaccard_index

import src.utils as utils
from src.models.metrics import compute_metrics
from src.models.model_loader import ModelLoader


class mtlMayhemModule(pl.LightningModule):
    def __init__(self, config_path, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        # load configuration scrip
        self.config = utils.load_yaml(config_path)
        self.class_lookup = utils.load_yaml("configs/class_lookup.yaml")

        # initialize model loader
        self.model_loader = ModelLoader(self.config)
        if self.config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3", "ssdlite"]:
            self.val_metric = "map"
        elif self.config["model"] in ["deeplabv3"]:
            self.val_metric = "miou"

        # check if config file is for trained model or not
        if utils.check_if_model_timestamped(config_path):
            # load name and paths from config file
            self.model_name = str(Path(config_path).stem)

            self.path_dict = utils.create_paths(
                model_name=self.model_name,
                model_folder=self.config["model_out_path"],
            )

        else:
            # create timestamped name for model
            self.model_name = utils.model_timestamp(model_name=self.config["model"], attribute=self.config["attribute"])

            # create paths for training
            self.path_dict = utils.create_paths(
                model_name=self.model_name, model_folder=self.config["model_out_path"], assert_paths=False
            )

            if not self.config["debug"]:
                # create folders for weights and checkpoints
                utils.create_model_folders(
                    config_old_path=config_path,
                    manifest_old_path=self.config["data_root"] + "/manifest.json",
                    path_dict=self.path_dict,
                )

        logging.info("Model instance ID: {}".format(self.model_name))

    def setup(self, stage: str) -> None:

        # load model
        self.model = self.model_loader.grab_model()

        # initialize variables
        self.best_result = 0
        self.epoch = 0

        # turn of logging plots for testing or validation
        if stage == "test" or stage == "validate":
            self.config["logging"] = False

            if Path(self.path_dict["weights_path"]).exists():
                self.model.load_state_dict(torch.load(self.path_dict["weights_path"] + "/best.pth"))
            else:
                raise FileExistsError("No trained model found.")

        # update configuration of hyperparams in wandb
        if self.config["logging"]:
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
                self.optimizer, step_size=lr_config["step_size"], gamma=lr_config["gamma"]
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

        # format to [B,C,H,W] tensor
        if isinstance(images, tuple):
            images = self.tuple_of_tensors_to_tensor(images)

        ### model specific forward pass ###

        if self.config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3", "ssdlite"]:
            loss_dict = self.model(images, targets)
            train_loss = sum(loss for loss in loss_dict.values())

        elif self.config["model"] in ["deeplabv3"]:
            # targets only contain masks as torch.BoolTensor
            targets = self.tuple_of_tensors_to_tensor(targets)

            preds = self.model(images)
            activation = torch.nn.Sigmoid()
            preds = activation(preds["out"])
            train_loss = dice(
                preds=preds, target=targets, ignore_index=self.class_lookup["sseg"]["background"]
            ).requires_grad_()

        ### _endof model specific forward pass ###

        if self.config["logging"]:
            self.log("train_loss", train_loss, on_step=True, on_epoch=True, batch_size=self.config["batch_size"])

        return train_loss

    def on_train_epoch_start(self) -> None:
        # initialize epoch counter
        self.epoch += 1
        if self.config["logging"]:
            self.log("epoch_sanity", torch.as_tensor(self.epoch, dtype=torch.float32), on_epoch=True)

    def on_validation_start(self) -> None:
        self.val_targets = []
        self.val_preds = []
        self.val_losses = []

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        """forward pass and loss calculation
        images: [B,C,H,W]
        targets: dictionary with keys "boxes", "masks", "labels"
        """
        images, targets = batch

        # format to [B,C,H,W] tensor
        if isinstance(images, tuple):
            images = self.tuple_of_tensors_to_tensor(images)

        if self.config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3", "ssdlite"]:
            preds = self.model(images)

        elif self.config["model"] in ["deeplabv3"]:
            # targets only contain masks as torch.BoolTensor
            targets = self.tuple_of_tensors_to_tensor(targets)

            preds = self.model(images)
            activation = torch.nn.Softmax()
            preds = activation(preds["out"])
            val_loss = jaccard_index(
                preds=preds,
                target=targets,
                num_classes=self.config["segmentation_classes"],
                ignore_index=self.class_lookup["sseg"]["background"],
            )
            if self.config["logging"]:
                self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.val_targets.extend(list(targets))
        self.val_preds.extend(preds)
        self.val_losses.append(val_loss)

    def on_validation_epoch_end(self) -> None:
        # skip sanity check
        if self.epoch > 0:

            if self.config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3", "ssdlite"]:
                # compute metrics
                results = compute_metrics(preds=self.val_preds, targets=self.val_targets)

                # extract mAP overall and for each class
                results_map = results["map"].item()
                classes_map = results["map_per_class"].tolist()

                results_classes_map = {
                    self.class_lookup["bbox_rev"][idx + 1]: map for idx, map in enumerate(classes_map)
                }

                if self.config["logging"]:
                    self.log("val_{}".format(self.val_metric), results_map, on_epoch=True)
                    self.log("class_{}".format(self.val_metric), results_classes_map, on_epoch=True)

                self._save_model(result=results["map"], save_on="max")

                logging.info("Current validation mAP: {:.6f}".format(results["map"]))
                logging.info("Best validation mAP: {:.6f}".format(self.best_result))

            elif self.config["model"] in ["deeplabv3"]:
                mIoU = torch.mean(torch.stack(self.val_losses))
                if self.config["logging"]:
                    self.log("val_{}".format(self.val_metric), mIoU, on_epoch=True)
                self._save_model(result=mIoU, save_on="max")

                logging.info("Current validation mIoU: {:.6f}".format(mIoU))
                logging.info("Best validation mIoU: {:.6f}".format(self.best_result))

    def _save_model(self, result, save_on: str):
        save_model = False
        if save_on == "max":
            save_model = self.best_result < result
        if save_on == "min":
            save_model = self.best_result > result

        if save_model:
            self.best_result = result
            if not self.config["debug"]:
                logging.info("Saving model weights.")
                torch.save(self.model.state_dict(), self.path_dict["weights_path"] + "/best.pth")
            else:
                logging.warning("DEBUG MODE: model weights are not saved")

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

        self.log("Val_Image", img, on_epoch=True)

    @staticmethod
    def tuple_of_tensors_to_tensor(tuple_of_tensors):
        return torch.stack(list(tuple_of_tensors), dim=0)
