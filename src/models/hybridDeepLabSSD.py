from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead, _mobilenet_extractor
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class HybridModel(SSD):
    def __init__(self, config):
        # pure mobilenet backbone
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            progress=True,
            norm_layer=norm_layer,
            reduced_tail=False,
        )

        # backbone for detection
        detection_backbone = _mobilenet_extractor(
            backbone, trainable_layers=6, norm_layer=norm_layer
        )  # NOTE: extacts backbone.features within

        # backbone for segmentation
        segmentation_backbone = IntermediateLayerGetter(backbone.features, return_layers={"16": "out"})
        segmentation_head = DeepLabHead(
            in_channels=backbone.features[16].out_channels,
            num_classes=config["segmentation_classes"] - 1,
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

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        return {
            "detection": super().forward(images, targets),
            "segmentation": self.segmentation_forward(images),
        }

    def segmentation_forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.segmenation_backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        return result
