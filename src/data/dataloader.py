import argparse
import logging

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset
from torchvision.io import read_image

import src.utils as utils
from src.data.manifests import generate_manifest


class mtlDataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = utils.load_yaml(config)

        logging.info("generating manifest for data from {}".format(self.config["data_root"]))
        self.manifests = generate_manifest(self.config["collections"], self.config["data_root"])

    def prepare_data(self) -> None:
        # split manifest file
        logging.info(
            "splitting dataset to train: {}% valid: {}% test: {}%".format(
                self.config["split_ratio"][0] * 100,
                self.config["split_ratio"][1] * 100,
                self.config["split_ratio"][2] * 100,
            )
        )
        self.train_split, self.valid_split, self.test_split = random_split(self.manifests, self.config["split_ratio"])

        # specific loaders for specific formats models take
        if self.config["model"] == "fasterrcnn":
            self.datasetObject = FasterRCNNDataset
        else:
            # custom for self-built models
            self.datasetObject = CustomDataset
        logging.info("Loading dataset object -> {}".format(self.datasetObject))

    def setup(self, stage) -> None:
        # FIXME: once we have proper test set annotated -> [train, valid]
        if stage == "fit":
            self.train_dataset = self.datasetObject(self.train_split)
            self.valid_dataset = self.datasetObject(self.valid_split)
        if stage == "validate":
            self.valid_dataset = self.datasetObject(self.valid_split)
        if stage == "test":
            self.test_dataset = self.datasetObject(self.test_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=self.config["shuffle"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=self.config["shuffle"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=self.config["shuffle"],
        )

    # @staticmethod
    # def collate_fn(batch):
    #     images, masks, boxes, labels =[],[],[],[]

    #     for image, mask, target in batch:
    #         images.append(image)
    #         masks.append(mask)
    #         boxes.append(target["boxes"])
    #         labels.append(target["labels"])

    #         targets={
    #             "boxes":torch.stack(boxes),
    #             "labels":torch.stack(labels)
    #         }

    #     return images, masks, targets

    @staticmethod
    def collate_fn(batch):
        """
        To handle the data loading as different images may have different number
        of objects and to handle varying size tensors as well.
        """
        return tuple(zip(*batch))


class CustomDataset(Dataset):
    """
    dataloader for the self-collected warehouse dataset with cvat labels
    returns a tuple with the following
    - rgb image tensor [C, H, W]
    - mask tensor [C, H, W]
    - bbox list of tuples [(class, [corner points])]
    """

    def __init__(self, data_split: Subset):
        self.dataset = data_split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = read_image(self.dataset[index]["path"])
        mask = read_image(self.dataset[index]["mask"])
        bbox_tuple = [(bbox["class"], bbox["corners"]) for bbox in self.dataset[index]["bbox"]]
        return image, mask, bbox_tuple


class FasterRCNNDataset(Dataset):
    def __init__(self, data_split: Subset) -> None:
        self.dataset = data_split
        self.transforms = transforms.Compose([transforms.ToTensor()])
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.transforms(Image.open(self.dataset[index]["path"]))
        mask = self.transforms(Image.open(self.dataset[index]["mask"]))
        target = {
            "boxes": torch.FloatTensor([bbox["corners"] for bbox in self.dataset[index]["bbox"]]),
            "labels": torch.LongTensor([bbox["class"] for bbox in self.dataset[index]["bbox"]]),
        }
        return image, mask, target

    # def collate_fn(batch):
    #     return tuple(zip(*batch))


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
    data_module.prepare_data()
    data_module.setup(stage="validate")
    dataloader = data_module.val_dataloader()
    it = iter(dataloader)
    first = next(it)
    second = next(it)
    logging.info("{}, {} itt a vege fuss el vele".format(first, second))


if __name__ == "__main__":
    main()
