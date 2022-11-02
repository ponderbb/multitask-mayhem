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


class WarehouseMTLDataModule(pl.LightningDataModule):
    def __init__(self, arguments) -> None:
        super().__init__()
        self.args = arguments

        if Path(self.args.config).exists():
            self.config = utils.load_yaml(self.args.config)
        else:
            raise FileNotFoundError("Config file can not be found.")

        self.manifests = generate_manifest(
            self.config["collections"], self.config["data_root"]
        )

    def setup(self, stage) -> None:
        # FIXME: once we have proper test set annotated -> [train, valid]

        # split manifest file
        if self.config["dataset"] == "warehouse-nn":
            logging.info(
                "splitting dataset to train: {}% valid: {}% test: {}%".format(
                    self.config["split_ratio"][0]*100,
                    self.config["split_ratio"][1]*100,
                    self.config["split_ratio"][2]*100,
                )
            )
            train_split, valid_split, test_split = random_split(
                self.manifests, self.config["split_ratio"]
            )
            datasetObject = WarehouseMTL

        elif self.config["dataset"] == "synthetic-nn":
            raise NotImplementedError("Dataloader not implemented yet!")

        if stage == "fit":
            self.train_dataset = datasetObject(train_split)
        if stage == "validate":
            self.valid_dataset = datasetObject(valid_split)
        if stage == "test":
            self.test_dataset = datasetObject(test_split)

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
    data_module.setup(stage="validate")
    dataloader = data_module.val_dataloader()
    it = iter(dataloader)
    first = next(it)
    second = next(it)
    logging.info("itt a vege fuss el vele")


if __name__ == "__main__":
    main()
