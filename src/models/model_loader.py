import logging
from collections import OrderedDict
from functools import partial

import torch.nn as nn
from torch import Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.ssd import SSD, SSDScoringHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    MobileNetV3,
    mobilenet_v3_large,
)
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


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
        elif config["model"] == "maskrcnn":
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
            return "hybrid", ["miou", "map"]
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
            pretrained=True,
            weights="DEFAULT",
        )
        in_features = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)  # NOTE: stolened values from a blogpost
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

    @staticmethod
    def _load_maskrcnn(config):
        # fastercnn based on resnet50 backbone
        model = maskrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["segmentation_classes"])

        # maskrcnn based on resnet50 backbone
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, config["segmentation_classes"]
        )
        return model

    @staticmethod
    def _load_hybrid(config):

        # backbone from mobilenet with COCO weights
        weights_backbone = MobileNet_V3_Large_Weights.verify(
            weights_backbone
        )  # NOTE: this might just be the weights for the backbone
        backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)

        segmentation_head = DeepLabHead(960, config["segmentation_classes"] - 1)

        anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
        out_channels = det_utils.retrieve_out_channels(backbone, (320, 320))
        num_anchors = anchor_generator.num_anchors_per_location()

        return None


class HybridModel(nn.Module):
    def __init__(self, backbone: nn.Module, detection_head: nn.Module, segmentation_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

    def forward(self, x: Tensor):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        return result
