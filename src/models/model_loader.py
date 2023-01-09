import logging
from functools import partial

import torch.nn as nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from src.models.hybridDeepLabSSD import HybridModel


class ModelLoader:
    @classmethod
    def grab_model(cls, config: dict):
        logging.info(f"Loading model: {config['model']}")

        if config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3"]:
            model = cls._load_fastercnn(config)
        elif config["model"] == "ssdlite":
            model = cls._load_ssdlite(config)
        elif config["model"] == "deeplabv3":
            model = cls._load_deeplabv3(config)
        elif config["model"] == "maskrcnn":
            model = cls._load_maskrcnn(config)
        elif config["model"] == "hybrid":
            model = cls._load_hybrid(config)
        else:
            raise ValueError("Model not supported")

        # TODO: model summary, params, weights, etc

        return model

    def get_type(config):  # TODO: load losses for models also from here
        if config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3"]:
            return "detection", "map"
        elif config["model"] == "ssdlite":
            return "detection", "map"
        elif config["model"] == "deeplabv3":
            return "segmentation", "miou"
        elif config["model"] == "maskrcnn":
            return "segmentation", "miou"
        elif config["model"] == "hybrid":
            return "hybrid", "miou"
        else:
            raise ValueError("Model not supported")

    # DETECTION BASELINES #
    @staticmethod
    def _load_fastercnn(config):
        if config["model"] == "fasterrcnn":
            model = fasterrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
        elif config["model"] == "fasterrcnn_mobilenetv3":
            model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["detection_classes"])
        return model

    @staticmethod
    def _load_ssdlite(config):
        model = ssdlite320_mobilenet_v3_large(
            # pretrained=True,
            # weights="DEFAULT",
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1  # TODO: CHANGED
        )
        size = (640, 480)
        model.transform = GeneralizedRCNNTransform(
            min(size),
            max(size),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            size_divisible=1,
            fixed_size=size,
        )
        in_features = det_utils.retrieve_out_channels(model.backbone, (640, 480))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_features,
            num_classes=config["detection_classes"],
            num_anchors=num_anchors,
            norm_layer=norm_layer,
        )
        return model

    # SEGMENTATION BASELINES #
    @staticmethod
    def _load_deeplabv3(config):
        model = deeplabv3_mobilenet_v3_large(pretrained=True, weights="DEFAULT")
        model.classifier = DeepLabHead(960, config["segmentation_classes"] - 1)
        return model

    # Multi-task models #
    @staticmethod
    def _load_hybrid(config):
        return HybridModel(config)
