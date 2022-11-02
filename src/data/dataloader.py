import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import src.utils as utils
from src.data.manifests import generate_manifest


class WarehouseMTLDataModule(pl.LightningDataModule):
    def __init__(self, arguments) -> None:
        super().__init__()
        self.args = arguments

        if Path(self.args.config).exists():
            self.config = utils.load_yaml(self.args.config)
        else:
            raise FileNotFoundError("Config file can not be found.")

        # self.all_image_paths = utils.list_image_paths(self.config["collections"])
        self.manifests = generate_manifest(
            self.config["collections"], self.config["data_root"]
        )

        # # create directory and verify path for output masks
        # self.mask_folder_path = os.path.join(self.config["data_root"],"test_masks")
        # if Path(self.mask_folder_path).exists():
        #     os.rmdir(self.mask_folder_path)
        # else:
        #     os.makedirs(self.mask_folder_path)

    def setup(self) -> None:
        train_set, valid_set, test_set = random_split(
            self.manifests, self.config["split_ratio"]
        )
        # return super().setup(stage)

    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return super().train_dataloader()

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return super().val_dataloader()

    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     return super().test_dataloader()

    # class WarehouseMTL(Dataset):
    #     def __init__(self) -> None:
    #         super().__init__()
    #     def __len__(self):
    #         return None
    #     def __getitem__(self, index) -> T_co:
    # return super().__getitem__(index)


def main():

    utils.set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="configs/debug_dataloader.yaml",
        help="path to config file for dataloading",
    )
    args = parser.parse_args()
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_fmt,
        force=True,
        handlers=[
            logging.FileHandler(".logging/debug_dataloader.log", "w"),
            logging.StreamHandler(),
        ],
    )

    data_module = WarehouseMTLDataModule(args)
    data_module.setup()
    # dataloader = data_module.val_dataloader()


if __name__ == "__main__":
    main()
