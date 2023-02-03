import logging
import os
import random
import shutil

import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import draw_segmentation_masks

import src.utils as utils
from src.models.model_loader import ModelLoader
from src.visualization.draw_things import draw_bounding_boxes

TEST_SET = "data/test/2022-09-23-10-07-37/synchronized_l515_image/"
MODEL_NAME = "frcnn-hybrid_10v1_23-02-03T012617"
CLASS_LOOKUP = utils.load_yaml("configs/class_lookup.yaml")
OUTPUT_PATH = "reports/test/"
model_config = "models/{}/{}.yaml".format(MODEL_NAME, MODEL_NAME)
model_weights = "models/{}/weights/best.pth".format(MODEL_NAME)

utils.logging_setup(model_config)


class ImageDataset(Dataset):
    def __init__(self, test_set, downsample: int = None) -> None:
        super().__init__()
        test_set = utils.list_files_with_extension(test_set, ".png", "path")
        random.seed(42)
        if downsample:
            self.image_list = random.sample(test_set, downsample)
        else:
            self.image_list = test_set
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(self.image_list[idx]))
        return image.type(torch.FloatTensor)


def main():

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = utils.load_yaml(model_config)
    utils.logging_setup()

    img_dataset = ImageDataset(TEST_SET, downsample=10)

    test_set_dataloader = DataLoader(
        dataset=img_dataset,
        drop_last=True,
    )

    model = ModelLoader.grab_model(config=config)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))

    starter_pass, ender_pass, ender_draw = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    for i, image in enumerate(test_set_dataloader):
        model.eval()
        starter_pass.record()
        pred = model(image.to(device))
        ender_pass.record()
        # pred = pred[0]
        image = image.mul(255).type(torch.uint8)

        if "detection" in pred.keys():
            score_mask = pred["detection"][0]["scores"] > 0.5
            boxes = pred["detection"][0]["boxes"][score_mask]
            labels = pred["detection"][0]["labels"][score_mask]
            scores = pred["detection"][0]["scores"][score_mask]

            label_names = [CLASS_LOOKUP["bbox_rev"][label.item()] for label in labels]

            drawn_image = draw_bounding_boxes(image=image.squeeze(0), boxes=boxes, labels=label_names, scores=scores)

        if "segmentation" in pred.keys():
            mask = torch.sigmoid(pred["segmentation"]) > 0.5
            drawn_image = draw_segmentation_masks(drawn_image, mask.squeeze(0), alpha=0.5, colors="green")

        image_pil = T.ToPILImage()(drawn_image)
        ender_draw.record()
        torch.cuda.synchronize()
        logging.info(f"Pass time: {starter_pass.elapsed_time(ender_pass):.2f} ms")
        logging.info(f"Draw time: {starter_pass.elapsed_time(ender_draw):.2f} ms")

        image_pil.save(f"reports/test/{i}-{starter_pass.elapsed_time(ender_draw):.2f}.png")


if __name__ == "__main__":
    main()
