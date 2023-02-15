import copy
import logging
import os
import random
import shutil
import urllib.request

import cv2
import cv_bridge
import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from PIL import Image
from sensor_msgs.msg import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

import src.utils as utils
from src.models.model_loader import ModelLoader
from src.visualization.draw_things import draw_bounding_boxes

TEST_SET = "data/interim/2022-09-23-10-07-37/"
MODEL_NAME = "frcnn-hybrid_10v1_23-02-03T012617"
CLASS_LOOKUP = utils.load_yaml("configs/class_lookup.yaml")
MODEL_CONFIG = "models/{}/{}.yaml".format(MODEL_NAME, MODEL_NAME)
MODEL_WEIGHTS = "models/{}/weights/best.pth".format(MODEL_NAME)


def visualization(prediction: torch.tensor):
    image = image.mul(255).type(torch.uint8)

    if "detection" in prediction.keys():
        score_mask = prediction["detection"][0]["scores"] > 0.5
        boxes = prediction["detection"][0]["boxes"][score_mask]
        labels = prediction["detection"][0]["labels"][score_mask]
        scores = prediction["detection"][0]["scores"][score_mask]

        label_names = [CLASS_LOOKUP["bbox_rev"][label.item()] for label in labels]

        drawn_image = draw_bounding_boxes(image=image.squeeze(0), boxes=boxes, labels=label_names, scores=scores)

    if "segmentation" in prediction.keys():
        mask = torch.sigmoid(prediction["segmentation"]) > 0.5
        drawn_image = draw_segmentation_masks(drawn_image, mask.squeeze(0), alpha=0.5, colors="green")

    image_array = drawn_image.detach().cpu().numpy()

    return image_array


class ObjectDetector:
    confidence_cfg = None

    def __init__(self, video_topic=None, transform=None, model=None, device=None):
        self.bridge = cv_bridge.CvBridge()

        self.video_sub = rospy.Subscriber(video_topic, Image, self.video_callback, queue_size=1)

        self.transform = transform
        self.model = model.eval()
        self.device = device

        rospy.loginfo("Starting node")

    def video_callback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

        processed_img = self.preprocess(cv_img)

        output = self.deep_learning_module(processed_img)

        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", output)
        cv2.waitKey(1)

    def preprocess(self, img):
        cpy = copy.deepcopy(img)
        rgb = cv2.cvtColor(cpy, cv2.COLOR_BGR2RGB)

        return rgb

    def depth_colorize(self, depth):
        cmap3 = plt.cm.jet
        depth = np.squeeze(depth)
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        depth = 255 * cmap3(depth)  # [:, :, :]  # H, W, C
        return depth.astype("uint8")

    def deep_learning_module(self, img):
        # input_batch = self.transform(img).to(device)
        prediction = self.model(img.to(self.device))

        output = visualization(prediction=prediction)

        return output


def main():

    # set up logging
    utils.logging_setup(MODEL_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    cfg = utils.load_yaml(MODEL_CONFIG)

    model = ModelLoader.grab_model(config=cfg)
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))

    rospy.init_node("mtl_bb")

    follower = ObjectDetector(video_topic="/l515/color/image_raw", model=model, device=device)
    
    rospy.spin()

if __name__ == "__main__":
    main()
