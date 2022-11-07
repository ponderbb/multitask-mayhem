import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

import src.utils as utils
from src.data.manifests import generate_manifest


class mtlDataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = utils.load_yaml(config)

        logging.info("generating manifest for data from {}".format(self.config["data_root"]))
        self.manifests = generate_manifest(
            self.config["collections"], self.config["data_root"]
        )

    def prepare_data(self) -> None:
        # split manifest file
        if self.config["dataset"] == "warehouse-nn":
            logging.info(
                "splitting dataset to train: {}% valid: {}% test: {}%".format(
                    self.config["split_ratio"][0]*100,
                    self.config["split_ratio"][1]*100,
                    self.config["split_ratio"][2]*100,
                )
            )
            self.train_split, self.valid_split, self.test_split = random_split(
                self.manifests, self.config["split_ratio"]
            )
            self.datasetObject = WarehouseMTL

        elif self.config["dataset"] == "synthetic-nn":
            raise NotImplementedError("Dataloader not implemented yet!")
            # self.datasetObject = ...

    def setup(self, stage) -> None:
        # FIXME: once we have proper test set annotated -> [train, valid]
        if stage == "fit":
            self.train_dataset = self.datasetObject(self.train_split)
        if stage == "validate":
            self.valid_dataset = self.datasetObject(self.valid_split)
        if stage == "test":
            self.test_dataset = self.datasetObject(self.test_split)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False)


class WarehouseMTL(Dataset):
    """
    dataloader for the self-collected warehouse dataset with cvat labels
    returns a tuple with the following
    - rgb image tensor [C, H, W]
    - mask tensor [C, H, W]
    - bbox list of tuples [(class, [corner points])]
    """

    def __init__(self, data_split):
        self.dataset = data_split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = read_image(self.dataset[index]["path"])
        mask = read_image(self.dataset[index]["mask"])
        bbox_tuple = [
            (bbox["class"], bbox["corners"]) for bbox in self.dataset[index]["bbox"]
        ]
        return (image, mask, bbox_tuple)


def main():

    utils.set_seeds()

    # grab arguments 
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
        default=".logging/dataloader_debug.log",
        help="path to config file for training",
    )
    args = parser.parse_args()

    utils.logging_setup(args.debug, args.log)

    data_module = mtlDataModule(args.config)
    data_module.setup(stage="validate")
    dataloader = data_module.val_dataloader()
    it = iter(dataloader)
    first = next(it)
    second = next(it)
    logging.info("itt a vege fuss el vele")


if __name__ == "__main__":
    main()
