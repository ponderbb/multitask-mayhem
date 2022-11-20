import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

import src.utils as utils
from src.data.dataloader import mtlDataModule
from src.models.lightning_frame import mtlMayhemModule

# import logging

utils.set_seeds()

# grab arguments
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    default="configs/dummy_training_hpc.yaml",
    help="Path to pipeline configuration file",
)
args = parser.parse_args()

# set up output logging
utils.logging_setup(args.config)

# initialize data pipeline
lightning_datamodule = mtlDataModule(args.config)

# initialize training pipeline
lightning_module = mtlMayhemModule(args.config)

# initialize Weights & Biases experiment logging
if lightning_module.config["logging"]:
    logger = WandbLogger(
        name=lightning_module.model_name,
        project=lightning_module.config["wandb_project"],
        entity=lightning_module.config["entity"],
        save_dir=lightning_module.checkpoints_landing,
        log_model=False,
    )
else:
    logger = None

# initialize trainer object
trainer = pl.Trainer(
    logger=logger,
    accelerator="auto",
    devices=1,
    enable_checkpointing=not (lightning_datamodule.config["debug"]),
    default_root_dir=lightning_module.checkpoints_landing,
    max_epochs=lightning_module.config["max_epochs"],
)

# start training
trainer.fit(model=lightning_module, datamodule=lightning_datamodule)
