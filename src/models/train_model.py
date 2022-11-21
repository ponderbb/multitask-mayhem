import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
        save_dir=lightning_module.config["local_logs"],
        log_model=False,
    )
else:
    logger = None

# callbacks
lr_monitor = LearningRateMonitor(logging_interval="epoch")
es_config = lightning_module.config["early_stop"]
early_stop_callback = EarlyStopping(
    monitor="val_result", min_delta=es_config["delta"], patience=es_config["patience"], verbose=False, mode="max"
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_result",
    dirpath=lightning_module.checkpoints_landing,
    filename="{epoch:02d}-{val_result:.4f}",
    save_top_k=3,
    mode="max",
)

# initialize trainer object
trainer = pl.Trainer(
    logger=logger,
    accelerator="auto",
    devices=1,
    enable_checkpointing=not (lightning_datamodule.config["debug"]),
    default_root_dir=lightning_module.checkpoints_landing,
    max_epochs=lightning_module.config["max_epochs"],
    callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],
)

# start training
trainer.fit(model=lightning_module, datamodule=lightning_datamodule)
