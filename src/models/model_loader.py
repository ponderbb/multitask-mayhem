import logging
from functools import partial

import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.lraspp import LRASPPHead

from src.models.hybridDeepLabFasterRCNN import FRCNNHybridModel
from src.models.hybridDeepLabSSD import SSDLiteHybridModel

IMAGE_SIZE = (640, 480)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelLoader:
    @classmethod
    def grab_model(cls, config: dict):
        logging.info(f"Loading model: {config['model']}")

        if config["model"] in ["frcnn-resnet", "frcnn"]:
            model = cls._load_fastercnn(config)
        elif config["model"] == "ssdlite":
            model = cls._load_ssdlite(config)
        elif config["model"] == "deeplabv3":
            model = cls._load_deeplabv3(config)
        elif config["model"] == "lraspp":
            model = cls._load_lraspp(config)
        elif config["model"] == "ssdlite-hybrid":
            model = cls._load_ssdlite_hybrid(config)
        elif config["model"] == "frcnn-hybrid":
            model = cls._load_frcnn_hybrid(config)
        else:
            raise ValueError("Model not supported")

        # TODO: model summary, params, weights, etc
        # logging.debug(model)

        return model

    def get_type(config):  # TODO: load losses for models also from here
        if config["model"] in ["frcnn-resnet", "frcnn"]:
            metrics = {"detection": "map"}
            losses = {
                "detection": MeanAveragePrecision(iou_type="bbox", class_metrics=config["class_metrics"]),
                "segmentation": None,
            }
        elif config["model"] == "ssdlite":
            metrics = {"detection": "map"}
            losses = {
                "detection": MeanAveragePrecision(iou_type="bbox", class_metrics=config["class_metrics"]),
                "segmentation": None,
            }
        elif config["model"] in ["deeplabv3", "lraspp"]:
            metrics = {"segmentation": "miou"}
            losses = {"detection": None, "segmentation": torch.nn.BCEWithLogitsLoss()}
        elif config["model"] in ["ssdlite-hybrid", "frcnn-hybrid"]:
            metrics = {"detection": "map", "segmentation": "miou"}
            losses = {
                "detection": MeanAveragePrecision(iou_type="bbox", class_metrics=config["class_metrics"]),
                "segmentation": torch.nn.BCEWithLogitsLoss(),
            }
        else:
            raise ValueError("Model not supported")

        tasks = list(metrics.keys())

        return tasks, metrics, losses

    # DETECTION BASELINES #
    @staticmethod
    def _load_fastercnn(config):
        if config["model"] == "frcnn-resnet":
            model = fasterrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
        elif config["model"] == "frcnn":
            model = fasterrcnn_mobilenet_v3_large_320_fpn(
                # pretrained=True,
                # weights="DEFAULT",
                weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
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
        model.transform = GeneralizedRCNNTransform(
            min(IMAGE_SIZE),
            max(IMAGE_SIZE),
            IMAGENET_MEAN,
            IMAGENET_STD,
            size_divisible=1,
            fixed_size=IMAGE_SIZE,
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
        model = deeplabv3_mobilenet_v3_large(
            # pretrained=True,
            # weights="DEFAULT",
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1  # TODO: CHANGED
        )
        model.classifier = DeepLabHead(960, config["segmentation_classes"] - 1)
        return model

    @staticmethod
    def _load_lraspp(config):
        model = lraspp_mobilenet_v3_large(
            # pretrained=True,
            # weights="DEFAULT",
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1  # TODO: CHANGED
        )
        model.classifier = LRASPPHead(
            low_channels=40,  # from C2 found in _lraspp_mobilenetv3 method
            high_channels=960,  # from C5 found in _lraspp_mobilenetv3 method
            num_classes=config["segmentation_classes"] - 1,
            inter_channels=128,
        )
        return model

    # Multi-task models #
    @staticmethod
    def _load_ssdlite_hybrid(config):
        return SSDLiteHybridModel(config)

    @staticmethod
    def _load_frcnn_hybrid(config):
        return FRCNNHybridModel(config)
