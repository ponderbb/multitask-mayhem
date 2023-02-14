import argparse
import json
import logging

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset

import src.utils as utils
from src.data.manifests import generate_manifest
from src.models.model_loader import ModelLoader


class mtlDataModule(pl.LightningDataModule):
    def __init__(self, config_path) -> None:
        super().__init__()

        self.config = utils.load_yaml(config_path)

        utils.set_seeds()

        logging.info("Data root folder -> {}".format(self.config["data_root"]))
        if utils.check_if_model_timestamped(config_path):
            # load from manifest file
            raise NotImplementedError("Loading manifest for inference not implemented yet")
        else:
            self.manifests = generate_manifest(
                collections=self.config["collections"], data_root=self.config["data_root"], create_mask=False
            )

        self.model_type, _, _ = ModelLoader.get_type(self.config)

    def prepare_data(self) -> None:
        # split manifest file
        logging.info(
            "Splitting dataset to train: {}% valid: {}%".format(
                self.config["split_ratio"][0] * 100,
                self.config["split_ratio"][1] * 100,
            )
        )
        self.train_split, self.valid_split = random_split(self.manifests, self.config["split_ratio"])

        self.datasetObject = UniversalDataloader

        logging.info("Loading dataset object -> {}".format(self.datasetObject))

    def setup(self, stage) -> None:

        if stage == "fit":
            self.train_dataset = self.datasetObject(self.train_split, self._compose_transforms())
            self.valid_dataset = self.datasetObject(self.valid_split, self._compose_transforms(eval=True))

        if stage == "validate":
            self.valid_dataset = self.datasetObject(self.valid_split, self._compose_transforms(eval=True))

        if stage == "test":
            self.test_manifest = json.load(open(self.config["test_manifest"], "r"))

            self.test_dataset = self.datasetObject(self.test_manifest, self._compose_transforms(eval=True))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=self.config["shuffle"],
            drop_last=True,
        )

    def meta_dataloader(self):
        return iter(
            DataLoader(
                self.train_dataset,
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                collate_fn=self.collate_fn,
                shuffle=self.config["shuffle"],
                drop_last=True,
            )
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collate_fn,
            drop_last=False,
        )

    @staticmethod
    def collate_fn(batch):
        """
        To handle the data loading as different images may have different number
        of objects and to handle varying size tensors as well.
        """
        return tuple(zip(*batch))

    def _compose_transforms(self, eval=False):

        transforms_list = []

        if not eval:
            if self.config["vflip"]["apply"]:
                transforms_list.append(A.VerticalFlip(p=self.config["vflip"]["p"]))
            if self.config["hflip"]["apply"]:
                transforms_list.append(A.HorizontalFlip(p=self.config["hflip"]["p"]))
            if self.config["rotate"]["apply"]:
                transforms_list.append(
                    A.Rotate(
                        limit=self.config["rotate"]["limit"],
                        p=self.config["rotate"]["p"],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    )
                )
        if self.config["normalize"]["apply"]:
            transforms_list.append(
                A.Normalize(
                    mean=self.config["normalize"]["mean"],
                    std=self.config["normalize"]["std"],
                    max_pixel_value=self.config["normalize"]["max_pixel_value"],
                )
            )
        transforms_list.append(ToTensorV2())

        transforms = A.Compose(
            transforms_list, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
        )

        return transforms


class UniversalDataloader(Dataset):
    def __init__(self, data_split: Subset, transforms: A.Compose) -> None:
        self.dataset = data_split
        self.transforms = transforms
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if "bbox" in self.dataset[index].keys():
            boxes = [bbox["corners"] for bbox in self.dataset[index]["bbox"]]
            labels = [bbox["class"] for bbox in self.dataset[index]["bbox"]]
        else:
            boxes = []
            labels = []

        transformed = self.transforms(
            image=np.array(Image.open(self.dataset[index]["path"]), copy=True),
            mask=np.array(Image.open(self.dataset[index]["mask"]), copy=True),
            bboxes=boxes,
            class_labels=labels,
        )

        image = transformed["image"].div(255)
        masks = transformed["mask"].type(torch.BoolTensor).unsqueeze(0)

        if len(transformed["bboxes"]) == 0:
            # dummy bbox and label variables
            boxes = torch.empty(size=[0, 4], dtype=torch.float32)
            labels = torch.empty(size=[0], dtype=torch.int64)
        else:
            boxes = torch.FloatTensor(transformed["bboxes"])
            labels = torch.LongTensor(transformed["class_labels"])

        target = {
            "boxes": boxes,  # N x 4
            "labels": labels,  # N
            "masks": masks,  # C x H x W
        }

        return image, target


def main():

    utils.set_seeds()

    # grab arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="configs/debug_foo.yaml",
        help="Path to pipeline configuration file",
    )
    args = parser.parse_args()

    utils.logging_setup(args.config)

    data_module = mtlDataModule(args.config)
    data_module.prepare_data()
    data_module.setup(stage="test")
    # dataloader = data_module.val_dataloader()
    # it = iter(dataloader)
    # first = next(it)
    # second = next(it)


if __name__ == "__main__":
    main()
