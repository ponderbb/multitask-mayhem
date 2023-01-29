from torch import nn, Tensor
from typing import Callable, Union, Dict
from collections import OrderedDict
from torchvision.utils import _log_api_usage_once
from torchvision.models.detection.ssdlite import _extra_block, _normal_init


class SSDLiteFeatureExtractorMobileNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        c4_pos: int,
        norm_layer: Callable[..., nn.Module],
        c2_pos: int = None,
        width_mult: float = 1.0,
        min_depth: int = 16,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if backbone[c4_pos].use_res_connect:
            raise ValueError("backbone[c4_pos].use_res_connect should be False")

        if c2_pos:
            self.features = nn.Sequential(
                # Extension to accomodate the feature extraction for LR-ASPP backbone
                nn.Sequential(*backbone[:c2_pos], backbone[c2_pos]),  # from start until C2 expansion layer
                nn.Sequential(backbone[c2_pos + 1:c4_pos], backbone[c4_pos].block[0]),  # from start until C4 expansion layer
                nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1 :]),  # from C4 depthwise until end
            )
        else: 
            self.features = nn.Sequential(
                # As described in section 6.3 of MobileNetV3 paper
                nn.Sequential(*backbone[:c4_pos], backbone[c4_pos].block[0]),  # from start until C4 expansion layer
                nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1 :]),  # from C4 depthwise until end
            )

        get_depth = lambda d: max(min_depth, int(d * width_mult))  # noqa: E731
        extra = nn.ModuleList(
            [
                _extra_block(backbone[-1].out_channels, get_depth(512), norm_layer),
                _extra_block(get_depth(512), get_depth(256), norm_layer),
                _extra_block(get_depth(256), get_depth(256), norm_layer),
                _extra_block(get_depth(256), get_depth(128), norm_layer),
            ]
        )
        _normal_init(extra)

        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Get feature maps from backbone and extra. Can't be refactored due to JIT limitations.
        output = []
        for block in self.features:
            x = block(x)
            output.append(x)

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def _mobilenet_extractor(
    backbone,
    trainable_layers: int,
    norm_layer: Callable[..., nn.Module],
    c2_bool: bool = False,
):
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    if c2_bool:
        lraspp_low_pos = stage_indices[-4] # C4 expansion layer plus one as numbering starts at 0
    else:
        lraspp_low_pos = None

    # find the index of the layer from which we wont freeze
    if not 0 <= trainable_layers <= num_stages:
        raise ValueError("trainable_layers should be in the range [0, {num_stages}], instead got {trainable_layers}")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDLiteFeatureExtractorMobileNet(
        backbone = backbone,
        c2_pos = lraspp_low_pos,
        c4_pos = stage_indices[-2],
        norm_layer = norm_layer
    )