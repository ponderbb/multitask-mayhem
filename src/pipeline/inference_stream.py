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

import pyrealsense2 as rs
import numpy as np
import cv2


TEST_SET = "data/interim/2022-09-23-10-07-37/"
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

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    cfg = utils.load_yaml(model_config)
    utils.logging_setup()

    # img_dataset = ImageDataset(TEST_SET)

    # test_set_dataloader = DataLoader(
    #     dataset=img_dataset,
    #     drop_last=True,
    # )

    model = ModelLoader.grab_model(config=cfg)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # starter_pass, ender_pass, ender_draw = (
    #     torch.cuda.Event(enable_timing=True),
    #     torch.cuda.Event(enable_timing=True),
    #     torch.cuda.Event(enable_timing=True),
    # )


        
        # ender_draw.record()
        # torch.cuda.synchronize()
        # logging.info(i)
        # logging.info(f"Pass time: {starter_pass.elapsed_time(ender_pass):.2f} ms")
        # logging.info(f"Draw time: {starter_pass.elapsed_time(ender_draw):.2f} ms")

        # image_pil.save(f"reports/test/{i}.png") # -{starter_pass.elapsed_time(ender_draw):.2f}.png")

    # Start streaming
    pipeline.start(config)

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))


            model.eval()
            # starter_pass.record()
            pred = model(image.to(device))
            # ender_pass.record()
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
            

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', image_pil)
            cv2.waitKey(1)
            
    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
