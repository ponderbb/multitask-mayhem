
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

import pytorch_lightning as pl
import src.utils as utils
from src.data.dataloader import mtlDataModule
from src.pipeline.lightning_frame import mtlMayhemModule

def main():

    # import logging

    utils.set_seeds()

    # grab arguments
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="models/emp_models/frcnn-hybrid_2v1_23-02-21T182529/frcnn-hybrid_2v1_23-02-21T182529.yaml",
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "-l",
        "--list",
        default=None
    )
    args = parser.parse_args()

    if args.list:
        runs_df = pd.read_csv(args.list, index_col=0)
        runs_list = runs_df["name"].values.tolist()
        folder = Path(args.list).parents[0]

        for run in runs_list:
            if not Path(f"reports/test_imgs/{run}").exists():
                logging.info(f"RUN: {run}")
                config = f"{folder}/{run}/{run}.yaml"
                run_inference(config)
            else:
                logging.info(f"RUN EXISTS: {run}")
    
    else:
        run_inference(args.config)


def run_inference(config):
    # set up output logging
    utils.logging_setup(config)

    utils.change_paths_in_config(
        input_yaml = config,
        out_yaml = config,
        # data_root = "data/interim",
        # test_manifest = "data/test/manifest.json",
        # model_out_path = "models/emp_models",
        class_metrics = True
    )

    # initialize data pipeline
    lightning_datamodule = mtlDataModule(config, test=True)

    # initialize training pipeline
    lightning_module = mtlMayhemModule(config)

    # TRAINER #
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_checkpointing=not (lightning_datamodule.config["debug"]),
        default_root_dir=lightning_module.path_dict["checkpoints_path"],
    )

    # start training
    trainer.test(model=lightning_module, datamodule=lightning_datamodule)

if __name__ == "__main__":
    main()