import copy

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.functional.classification import binary_jaccard_index

import src.utils as utils
from src.features.metrics import compute_metrics
from src.pipeline.lightning_utils import plUtils


class AutoLambda:
    def __init__(self, model, device, train_tasks, pri_tasks, weight_init=0.1):
        self.model = model
        self.model_ = copy.deepcopy(model)
        self.meta_weights = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.device = device
        self.class_lookup = utils.load_yaml("configs/class_lookup.yaml")

    def virtual_step(self, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        train_pred = self.model(train_x, train_y)

        # sum detection bbox_regression and cross-entropy
        train_pred["detection"] = sum(loss for loss in train_pred["detection"].values())

        train_loss_list = list(train_pred.values())

        loss = sum([w * train_loss_list[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if "momentum" in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get("momentum_buffer", 0.0) * model_optim.param_groups[0]["momentum"]
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]["weight_decay"] * weight))

    def unrolled_backward(self, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        # compute validation data loss on primary tasks
        # val_pred = self.model_(val_x)
        # val_loss = self.model_fit(val_pred, val_y)

        val_pred = self.model_(val_x, val_y)

        # sum detection bbox_regression and cross-entropy
        val_pred["detection"] = sum(loss for loss in val_pred["detection"].values())

        val_loss_list = list(val_pred.values())

        loss = sum([w * val_loss_list[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = -alpha * h

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        train_pred = self.model(train_x, train_y)
        # sum detection bbox_regression and cross-entropy
        train_pred["detection"] = sum(loss for loss in train_pred["detection"].values())
        train_loss_list = list(train_pred.values())
        loss = sum([w * train_loss_list[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        train_pred = self.model(train_x, train_y)
        # sum detection bbox_regression and cross-entropy
        train_pred["detection"] = sum(loss for loss in train_pred["detection"].values())
        train_loss_list = list(train_pred.values())
        loss = sum([w * train_loss_list[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2.0 * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit(self, pred, targets):
        """
        define task specific losses
        """
        target_masks = plUtils.tuple_of_tensors_to_tensor(tuple([target["masks"] for target in targets]))

        det_metrics = compute_metrics(
            metric_box=MeanAveragePrecision(iou_type="bbox"),
            preds=pred["detection"],
            targets=targets,
        )

        seg_metrics = binary_jaccard_index(
            preds=pred["segmentation"],
            target=target_masks.type(torch.float32).to(self.device),
            ignore_index=self.class_lookup["sseg"]["background"],
        )

        return [det_metrics["map"].requires_grad_().to(self.device), seg_metrics.requires_grad_()]
