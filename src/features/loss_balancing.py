import logging

import torch

from src.features.autolambda import AutoLambda
from src.pipeline.lightning_utils import plUtils


class LossBalancing:
    def __init__(self, config: dict, model_tasks: list[str], logger):
        self.config = config
        self.model_tasks = model_tasks
        self.task_count = len(model_tasks)
        self.log = logger
        self.nMinus1_loss = []
        self.nMinus2_loss = []

        if self.task_count > 1:
            if self.config["weight"] == "uncertainty":
                weights_init_tensor = torch.Tensor([-0.7] * self.task_count)
                self.logsigma = torch.nn.parameter.Parameter(weights_init_tensor, requires_grad=True)

            elif self.config["weight"] in ["dynamic", "equal"]:
                self.temperature = 2.0
                self.lambda_weight = torch.ones(self.task_count)

            elif self.config["weight"] == "constant":
                self.lambda_weight = torch.Tensor(self.config["w_constant"])

            elif self.config["weight"] == "autol":
                self.autol = AutoLambda(self.model, self.device, self.model_tasks, self.model_tasks)
                self.meta_optimizer = torch.optim.Adam([self.autol.meta_weights], lr=self.config["optimizer"]["lr"])

            else:
                raise ValueError("Unknown loss balancing method.")
        else:
            logging.info("Single task detected, no loss weighting applied.")

            return None

    def calculate_lambda_weight(self, epoch: int):
        if epoch > 2:
            w = []
            for n1, n2 in zip(self.nMinus1_loss, self.nMinus2_loss):
                w.append(n1 / n2)
            w = torch.softmax(torch.tensor(w) / self.temperature, dim=0)
            self.lambda_weight = self.task_count * w.numpy()

    def loss_balancing_step(self, train_loss: dict):

        train_loss_list = list(train_loss.values())

        if self.config["weight"] == "uncertainty":
            balanced_loss = [1 / (2 * torch.exp(w)) * train_loss_list[i] + w / 2 for i, w in enumerate(self.logsigma)]

            if self.config["logging"]:
                self.log(
                    "logsigma",
                    {"det": self.logsigma[0], "seg": self.logsigma[1]},
                    on_epoch=True,
                )

        if self.config["weight"] in ["equal", "constant", "dynamic"]:
            balanced_loss = [w * train_loss_list[i] for i, w in enumerate(self.lambda_weight)]

            if self.config["logging"]:
                self.log(
                    "lambda_weight",
                    {"det": self.lambda_weight[0], "seg": self.lambda_weight[1]},
                    on_epoch=True,
                )

        if self.config["weight"] == "autol":
            balanced_loss = [w * train_loss_list[i] for i, w in enumerate(self.autol.meta_weights)]

        return {"master": sum(balanced_loss), "det": balanced_loss[0], "seg": balanced_loss[1]}

    def update_meta_weights(self, train_image, train_target):
        if self.config["weight"] == "autol":
            val_image, val_target = self.trainer.datamodule.meta_dataloader()._next_data()

            if isinstance(val_image, tuple):
                val_image = plUtils.tuple_of_tensors_to_tensor(val_image).to(self.device)

            self.meta_optimizer.zero_grad()
            self.autol.unrolled_backward(
                train_image, train_target, val_image, val_target, self.lr_scheduler.get_last_lr()[0], self.optimizer
            )

            self.meta_optimizer.step()

    def loss_balancing_epoch_end(self, train_loss: dict):
        if self.config["weight"] == "dynamic":

            train_loss_detached = {k: v.detach() for k, v in train_loss.items()}
            train_loss_detached.pop("master")

            if self.epoch == 1:
                self.nMinus2_loss = list(train_loss_detached.values())
            elif self.epoch == 2:
                self.nMinus1_loss = list(train_loss_detached.values())
            else:
                self.nMinus2_loss = self.nMinus1_loss
                self.nMinus1_loss = list(train_loss_detached.values())
