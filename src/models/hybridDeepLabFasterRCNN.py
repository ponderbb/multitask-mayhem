import warnings
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from src.pipeline.lightning_utils import plUtils


class FRCNNHybridModel(FasterRCNN):
    def __init__(self, config):
        # pure mobilenet backbone
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            progress=True,
            norm_layer=norm_layer,
            reduced_tail=False,
        )

        fastercnn_backbone = _mobilenet_extractor(backbone=backbone, fpn=True, trainable_layers=6)

        anchor_sizes = (
            (
                32,
                64,
                128,
                256,
                512,
            ),
        ) * 3
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        super().__init__(
            backbone=fastercnn_backbone,
            num_classes=config["detection_classes"],
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
            # min_size=480, #BUG: wrong aspectratio
            # max_size=640,
        )

        # backbone for segmentation
        self.segmentation_head = DeepLabHead(
            in_channels=960,
            num_classes=config["segmentation_classes"] - 1,
        )

        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.shared_backbone = [fastercnn_backbone]

    def shared_modules(self):
        return self.shared_backbone

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
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

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features_deeplab = self.backbone._modules["body"](images.tensors)
        features = self.backbone._modules["fpn"](features_deeplab)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        seg_output = self.segmentation_decoder(features=features_deeplab["1"], input_shape=original_image_sizes[0])

        if self.training:
            # output losses just like the detection module, if we provide targets
            target_masks = tuple([target["masks"] for target in targets_original])
            target_masks = plUtils.tuple_of_tensors_to_tensor(target_masks)
            seg_output = self.segmentation_loss(seg_output, target_masks.type(torch.float32))

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return {"detection": self.eager_outputs(losses, detections), "segmentation": seg_output}

    def segmentation_decoder(self, features, input_shape):
        x = self.segmentation_head(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x
