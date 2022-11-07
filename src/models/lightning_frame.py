import pytorch_lightning as pl
import torch
import wandb
from typing import Any, Optional
import src.utils as utils

from torchvision.models.detection import fasterrcnn_resnet50_fpn

import os
from pathlib import Path

class mtlMayhemModule(pl.LightningModule):
    def __init__(self, config, ) -> None:
        super().__init__()
        self.config = utils.load_yaml(config)

    def setup(self, stage: str) -> None:
        # turn of logging plots for testing or validation
        if stage == "test" or stage == "validate":
            self.config["logging"] = False

        # update configuration of hyperparams in wandb
        if self.config["logging"]:
            wandb.config.update(self.config)

        if self.config["model"] == "fasterrcnn":
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        return super().setup(stage)

    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()

    def training_step(self, *args: Any, **kwargs: Any):
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)