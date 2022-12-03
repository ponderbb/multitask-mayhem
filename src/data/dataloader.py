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

    def prepare_data(self) -> None:
        # split manifest file
        logging.info(
            "Splitting dataset to train: {}% valid: {}% test: {}%".format(
                self.config["split_ratio"][0] * 100,
                self.config["split_ratio"][1] * 100,
                self.config["split_ratio"][2] * 100,
            )
        )
        self.train_split, self.valid_split, self.test_split = random_split(self.manifests, self.config["split_ratio"])

        # specific loaders for specific formats models take
        if self.config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3", "ssdlite"]:
            self.datasetObject = FasterRCNNDataset
        elif self.config["model"] in ["deeplabv3"]:
            self.datasetObject = DeepLabV3Dataset
        else:
            raise NotImplementedError("Dataloader could not be found for {}".format(self.config["model"]))

        logging.info("Loading dataset object -> {}".format(self.datasetObject))

    def setup(self, stage) -> None:
        if stage == "fit":
            self.train_dataset = self.datasetObject(self.train_split)
            self.valid_dataset = self.datasetObject(self.valid_split)

        if stage == "validate":
            self.valid_dataset = self.datasetObject(self.valid_split)

        if stage == "test":  # TODO: split rest to [train, test]
            self.test_dataset = self.datasetObject(self.test_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=self.config["shuffle"],
            drop_last=True,
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
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=True,
        )

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

        # check for images without bounding boxes
        if "bbox" in self.dataset[index].keys():
            boxes = torch.FloatTensor([bbox["corners"] for bbox in self.dataset[index]["bbox"]])
            labels = torch.LongTensor([bbox["class"] for bbox in self.dataset[index]["bbox"]])
        else:
            boxes = torch.empty(size=[0, 4], dtype=torch.float32)
            labels = torch.empty(size=[0], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": torch.as_tensor(mask, dtype=torch.bool),
        }

        return image, target


class DeepLabV3Dataset(Dataset):
    def __init__(self, data_split: Subset) -> None:
        self.dataset = data_split
        self.transforms = transforms.Compose([transforms.ToTensor()])
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.transforms(Image.open(self.dataset[index]["path"]))
        mask = read_image(self.dataset[index]["mask"]).type(torch.BoolTensor)

        return image, mask


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
    args = parser.parse_args()

    utils.logging_setup(args.config)

    data_module = mtlDataModule(args.config)
    data_module.prepare_data()
    data_module.setup(stage="validate")
    # dataloader = data_module.val_dataloader()
    # it = iter(dataloader)
    # first = next(it)
    # second = next(it)


if __name__ == "__main__":
    main()
