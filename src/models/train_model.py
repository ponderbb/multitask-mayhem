import src.utils as utils
import logging
import argparse
from src.models.lightning_frame import mtlMayhemModule
from src.data.dataloader import mtlDataModule
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl

utils.set_seeds()

# grab arguments 
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    default="configs/dummy_training.yaml",
    help="Path to pipeline configuration file",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Include debug level information in the logging.",
)
parser.add_argument(
    "-l",
    "--log",
    default=".logging/training_pipeline.log",
    help="path to config file for training",
)
args = parser.parse_args()

# set up output logging
utils.logging_setup(args.debug, args.log)

# initialize data pipeline
lightning_datamodule = mtlDataModule(args.config)

# initialize training pipeline 
lightning_module = mtlMayhemModule(args.config)

# initialize Weights & Biases experiment logging
if lightning_module.config["logging"]:
    logger = WandbLogger() # TODO: setup team and project
else:
    logger = None

# initialize trainer object
trainer = pl.Trainer(
    logger=logger
)

if args.debug:
    # run validation hook to test I/Os
    trainer.validate(model=lightning_module, datamodule=lightning_datamodule)
else:
    # start training
    trainer.fit(model=lightning_module, datamodule=lightning_datamodule)