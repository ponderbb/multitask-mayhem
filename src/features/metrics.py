from typing import Dict, List

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def compute_metrics(
    preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Compute metrics for the model"""
    metric_box = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric_box.update(preds, targets)
    return metric_box.compute()
