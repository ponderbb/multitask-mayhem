from typing import Dict, List

import torch


def compute_metrics(
    metric_box,
    preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Compute metrics for the model"""
    metric_box.update(preds, targets)
    return metric_box.compute()
