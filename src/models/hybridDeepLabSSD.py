import warnings
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.lraspp import LRASPPHead
from torchvision.ops import boxes as box_ops

from src.models.SSD_utils import _mobilenet_extractor
from src.pipeline.lightning_utils import plUtils

# IMAGE_SIZE = (640, 480)
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


class SSDLiteHybridModel(SSD):
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
            backbone,
            trainable_layers=6,
            norm_layer=norm_layer,
            c2_bool=True if config["model"] == "lraspp-hybrid" else False,
        )  # constructs the extra SSD features on top of the mobilenet

        # detection head
        size = (640, 480)
        anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
        num_anchors = anchor_generator.num_anchors_per_location()
        in_features = det_utils.retrieve_out_channels(detection_backbone, size=size)
        if config["model"] == "lraspp-hybrid":
            in_features = in_features[1:]
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

        # backbone for segmentation
        if config["model"] == "ssdlite-hybrid":
            self.segmentation_head = DeepLabHead(
                in_channels=detection_backbone.features[1]._modules["3"].out_channels,
                num_classes=config["segmentation_classes"] - 1,
            )
            self.lraspp_mode = False

        elif config["model"] == "lraspp-hybrid":
            self.segmentation_head = LRASPPHead(
                low_channels=detection_backbone.features[0]._modules["4"].out_channels,
                high_channels=detection_backbone.features[2]._modules["3"].out_channels,
                num_classes=config["segmentation_classes"] - 1,
                inter_channels=128,
            )
            self.lraspp_mode = True

        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.shared_backbone = [detection_backbone]

    def shared_modules(self):
        return self.shared_backbone

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

            targets_original = targets

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        if self.lraspp_mode:
            lraspp_low_features = features["0"]
            lraspp_high_features = features["2"]
            features.pop("0")

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            matched_idxs = []
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for anchors_per_image, targets_per_image in zip(anchors, targets):
                    if targets_per_image["boxes"].numel() == 0:
                        matched_idxs.append(
                            torch.full(
                                (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                            )
                        )
                        continue

                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    matched_idxs.append(self.proposal_matcher(match_quality_matrix))

                losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections

        if self.lraspp_mode:
            seg_output = self.lraspp_decoder(
                low_features=lraspp_low_features,
                high_features=lraspp_high_features,
                input_shape=original_image_sizes[0],
            )
        else:
            seg_output = self.deeplabv3_decoder(features=features[1], input_shape=original_image_sizes[0])

        if self.training:
            # output losses just like the detection module, if we provide targets
            target_masks = tuple([target["masks"] for target in targets_original])
            target_masks = plUtils.tuple_of_tensors_to_tensor(target_masks)
            seg_output = self.segmentation_loss(seg_output, target_masks.type(torch.float32))

        return {"detection": self.eager_outputs(losses, detections), "segmentation": seg_output}

    def lraspp_decoder(self, low_features, high_features, input_shape):
        x = self.segmentation_head.cbr(high_features)
        s = self.segmentation_head.scale(high_features)
        x = x * s
        x = F.interpolate(x, size=low_features.shape[-2:], mode="bilinear", align_corners=False)

        out = self.segmentation_head.low_classifier(low_features) + self.segmentation_head.high_classifier(x)

        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)

        return out

    def deeplabv3_decoder(self, features, input_shape):
        x = self.segmentation_head(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x
