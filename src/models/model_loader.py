import logging
from functools import partial

import torch.nn as nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class ModelLoader:
    def __init__(self, config: dict):
        self.config = config

    def grab_model(self):
        logging.info(f"Loading model: {self.config['model']}")

        if self.config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3"]:
            self._load_fastercnn()
        elif self.config["model"] == "ssdlite":
            self._load_ssdlite()
        elif self.config["model"] == "deeplabv3":
            self._load_deeplabv3()
        elif self.config["model"] == "maskrcnn":
            self._load_maskrcnn()
        else:
            raise ValueError("Model not supported")

        # TODO: model summary, params, weights, etc

        return self.model

    def get_type(self):  # TODO: load losses for models also from here
        if self.config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3"]:
            return "detection", "map"
        elif self.config["model"] == "ssdlite":
            return "detection", "map"
        elif self.config["model"] == "deeplabv3":
            return "segmentation", "miou"
        elif self.config["model"] == "maskrcnn":
            return "segmentation", "miou"
        else:
            raise ValueError("Model not supported")

    ### DETECTION BASELINES ###

    def _load_fastercnn(self):
        if self.config["model"] == "fasterrcnn":
            self.model = fasterrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
        elif self.config["model"] == "fasterrcnn_mobilenetv3":
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["detection_classes"])

    def _load_ssdlite(self):
        self.model = ssdlite320_mobilenet_v3_large(
            pretrained=True,
            weights="DEFAULT",
        )
        in_features = det_utils.retrieve_out_channels(self.model.backbone, (320, 320))
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)  # NOTE: stolened values from a blogpost
        self.model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_features,
            num_classes=self.config["detection_classes"],
            num_anchors=num_anchors,
            norm_layer=norm_layer,
        )

    ### SEGMENTATION BASELINES ###

    def _load_deeplabv3(self):
        self.model = deeplabv3_mobilenet_v3_large(pretrained=True, weights="DEFAULT")
        self.model.classifier = DeepLabHead(960, self.config["segmentation_classes"] - 1)

    def _load_maskrcnn(self):
        # fastercnn based on resnet50 backbone
        self.model = maskrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["segmentation_classes"])

        # maskrcnn based on resnet50 backbone
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, self.config["segmentation_classes"]
        )
