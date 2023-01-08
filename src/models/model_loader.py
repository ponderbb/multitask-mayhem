import logging
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import torch
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
from torchvision.models.detection.ssdlite import (
    SSDLiteClassificationHead,
    _mobilenet_extractor,
    SSDLiteHead
)
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    MobileNetV3,
    mobilenet_v3_large,
)
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops


class ModelLoader:
    @classmethod
    def grab_model(cls, config: dict):
        logging.info(f"Loading model: {config['model']}")

        if config["model"] in ["fasterrcnn", "fasterrcnn_mobilenetv3"]:
            model = cls._load_fastercnn(config)
        elif config["model"] == "ssdlite":
            model = cls._load_ssdlite_alter(config)
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
            # pretrained=True,
            # weights="DEFAULT",
            weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 #TODO: CHANGED
        )
        size = (640,480)
        model.transform = GeneralizedRCNNTransform(
            min(size), max(size), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], size_divisible=1, fixed_size=size
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



    @staticmethod
    def _load_ssdlite_alter(config):
        return HybridModel(config)

class HybridModel(SSD):
    def __init__(self, config):
        # pure mobilenet backbone
        # weights_backbone = MobileNet_V3_Large_Weights.verify(MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            progress=True,
            norm_layer = norm_layer,
            reduced_tail=False,
        )

        # backbone for detection
        detection_backbone = _mobilenet_extractor(
            backbone,
            trainable_layers=6,
            norm_layer=norm_layer
        ) # NOTE: extacts backbone.features within

        # backbone for segmentation
        segmentation_backbone = IntermediateLayerGetter(backbone.features, return_layers={"16": "out"})
        segmentation_head = DeepLabHead(
            in_channels = backbone.features[16].out_channels,
            num_classes=config["segmentation_classes"] - 1
        )

        # detection head
        size = (640, 480)
        anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
        num_anchors = anchor_generator.num_anchors_per_location()
        in_features = det_utils.retrieve_out_channels(detection_backbone, size=size)
        head = SSDLiteHead(
            in_channels=in_features,
            num_anchors=num_anchors,
            num_classes=config["detection_classes"],
            norm_layer=norm_layer,
        )

        super().__init__(
            backbone=detection_backbone,
            num_classes=config["detection_classes"],
            head=head,
            size=size,
            anchor_generator=anchor_generator,
        )


        self.segmenation_backbone = segmentation_backbone
        self.segmentation_head = segmentation_head

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        return super().forward(images, targets), self.segmentation_forward(images)

    def segmentation_forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        # seg_backbone = IntermediateLayerGetter(self.segmenation_backbone.features, return_layers={"16": "out"})
        seg_backbone = self.segmenation_backbone
        features = self.segmenation_backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        return result





    # SEGMENTATION BASELINES #
    @staticmethod
    def _load_deeplabv3(config):
        model = deeplabv3_mobilenet_v3_large(pretrained=True, weights="DEFAULT")
        model.classifier = DeepLabHead(960, config["segmentation_classes"] - 1)
        return model

#     @staticmethod
#     def _load_hybrid(config):
#         # backbone from mobilenet with COCO weights
#         weights_backbone = MobileNet_V3_Large_Weights.DEFAULT
#         backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
#         backbone = backbone.features

#         # segmentation head
#         out_pos = 16 # Convolution 5, with output stride 16
#         segmentation_backbone = IntermediateLayerGetter(backbone, return_layers={str(out_pos):"out"}) # TODO: might have to move to the forward pass
#         segmentation_head = DeepLabHead(
#             out_inplanes = backbone[16].out_channels,
#             num_channels =config["segmentation_classes"] - 1
#         )

#         detection_backbone = _mobilenet_extractor(
#             backbone=backbone
#         )  # extracting the correct feature layers for ssdlite
#         anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
#         out_channels = det_utils.retrieve_out_channels(detection_backbone, (320, 320))
#         num_anchors = anchor_generator.num_anchors_per_location()

#         detection_head = None

#         model = HybridModel(backbone=backbone, detection_head=detection_head, segmentation_head=segmentation_head)

#         return None


# class HybridModel(nn.Module):
#     def __init__(self, backbone: nn.Module, detection_head: nn.Module, segmentation_head: nn.Module) -> None:
#         super().__init__()
#         self.backbone = backbone
#         self.detection_head = detection_head
#         self.segmentation_head = segmentation_head

#     def forward(self, x: Tensor):
#         input_shape = x.shape[-2:]
#         # contract: features is a dict of tensors
#         features = self.backbone(x)

#         result = OrderedDict()
#         x = features["out"]
#         x = self.classifier(x)
#         x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
#         result["out"] = x

#         return result
