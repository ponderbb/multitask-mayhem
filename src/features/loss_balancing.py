import logging

import torch

from src.features.autolambda import AutoLambda
from src.pipeline.lightning_utils import plUtils


class LossBalancing:
    def __init__(
        self, config: dict, model, device, lr_scheduler, optimizer, model_tasks: list[str], meta_dataloader, logger
    ):
        self.config = config
        self.model_tasks = model_tasks
        self.model = model
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.task_count = len(model_tasks)
        self.log = logger
        self.meta_dataloader = meta_dataloader
        self.nMinus1_loss = []
        self.nMinus2_loss = []

        if self.task_count > 1:
            if self.config["weight"] == "uncertainty":
                weights_init_tensor = torch.Tensor([-0.7] * self.task_count)
                self.logsigma = torch.nn.parameter.Parameter(weights_init_tensor, requires_grad=True)
                self.optimizer.param_groups[0]["params"].append(self.logsigma)
                pass

            elif self.config["weight"] in ["dynamic", "equal", "bmtl"]:

                if self.config["weight"] != "equal":
                    self.temperature = torch.tensor(
                        self.config["temperature"], device=self.device
                    )  # defining the smoothening of the softmax function

                if self.config["w_constant"]:
                    self.lambda_weight = torch.tensor(self.config["w_constant"], device=self.device)
                else:
                    self.lambda_weight = torch.ones(self.task_count) / self.task_count

            elif self.config["weight"] == "relative":
                self.difficulty = torch.zeros(self.task_count)

                if self.config["w_constant"]:
                    self.lambda_weight = torch.tensor(self.config["w_constant"], device=self.device)
                else:
                    self.lambda_weight = torch.ones(self.task_count) / self.task_count

            elif self.config["weight"] == "constant":
                self.lambda_weight = torch.tensor(self.config["w_constant"], device=self.device)

            elif self.config["weight"] == "autol":
                self.autol = AutoLambda(self.model, self.device, self.model_tasks, self.model_tasks)
                self.meta_optimizer = torch.optim.Adam([self.autol.meta_weights], lr=self.config["optimizer"]["lr"])

            elif self.config["weight"] == "geometric":
                pass

            else:
                raise ValueError("Unknown loss balancing method.")
        else:
            logging.info("Single task detected, no loss weighting applied.")

            return None

    def calculate_lambda_weight(self, epoch: int):

        if epoch > 2:
            if self.config["weight"] == "dynamic":
                w = []
                for n1, n2 in zip(self.nMinus1_loss, self.nMinus2_loss):
                    w.append(n1 / n2)
                w = torch.softmax(torch.tensor(w, device=self.device) / self.temperature, dim=0)
                self.lambda_weight = self.task_count * w.cpu().numpy()

            elif self.config["weight"] == "relative":
                """
                Calculate exponential moving average of difficulty and relative difficulty
                X_hat = alpha * X_i + (1 - alpha) * X_imin1 where X= D or D_R
                from Wenwen et al. 2020 https://doi.org/10.1016/j.neucom.2020.11.024
                """

                difficulty_sum = torch.zeros(1, device=self.device)

                for task, (n1, n2) in enumerate(zip(self.nMinus1_loss, self.nMinus2_loss)):
                    if epoch > 3:
                        self.difficulty[task] = self._calculate_ema(
                            x_i=torch.exp((n1 - n2) / n2),
                            x_imin1=self.difficulty[task],
                            alpha=torch.tensor([0.7], device=self.device),
                        )
                    else:
                        self.difficulty[task] = torch.exp((n1 - n2) / n2)

                    difficulty_sum += self.difficulty[task]

                for task, task_difficulty in enumerate(self.difficulty):
                    if epoch > 3:
                        relative_difficulty = torch.div(task_difficulty, difficulty_sum)
                        self.lambda_weight[task] = self._calculate_ema(
                            x_i=relative_difficulty,
                            x_imin1=self.lambda_weight[task],
                            alpha=torch.tensor([0.7], device=self.device),
                        )
                    else:
                        self.lambda_weight[task] = torch.div(task_difficulty, difficulty_sum)

                    # self.lambda_weight[task] = self.lambda_weight[task] * torch.tensor([10.0], device=self.device)

    @staticmethod
    def _calculate_ema(x_i: torch.tensor, x_imin1: torch.tensor, alpha: torch.tensor):
        return alpha * x_i + (1 - alpha) * x_imin1

    def loss_balancing_step(self, train_loss: dict):

        train_loss_list = list(train_loss.values())

        if self.config["weight"] == "uncertainty":
            balanced_loss = [1 / (2 * torch.exp(w)) * train_loss_list[i] + w / 2 for i, w in enumerate(self.logsigma)]

            if self.config["logging"]:
                self.log(
                    "logsigma",
                    {"det": self.logsigma[0], "seg": self.logsigma[1]},
                    on_epoch=True,
                    on_step=True,
                    batch_size=self.config["batch_size"],
                )

        if self.config["weight"] in ["equal", "constant", "dynamic", "relative"]:
            balanced_loss = [w * train_loss_list[i] for i, w in enumerate(self.lambda_weight)]

            if self.config["logging"]:
                self.log(
                    "lambda_weight",
                    {"det": self.lambda_weight[0], "seg": self.lambda_weight[1]},
                    on_epoch=True,
                    on_step=True,
                    batch_size=self.config["batch_size"],
                )

        if self.config["weight"] == "autol":
            balanced_loss = [w * train_loss_list[i] for i, w in enumerate(self.autol.meta_weights)]

            if self.config["logging"]:
                self.log(
                    "meta_weight",
                    {"det": self.autol.meta_weights[0], "seg": self.autol.meta_weights[1]},
                    on_epoch=True,
                    on_step=True,
                    batch_size=self.config["batch_size"],
                )

        if self.config["weight"] == "geometric":
            balanced_loss = [torch.sqrt(task_loss) for task_loss in train_loss_list]
            return {"master": balanced_loss[0] * balanced_loss[1], "det": balanced_loss[0], "seg": balanced_loss[1]}

        if self.config["weight"] == "bmtl":
            balanced_loss = [torch.exp(task_loss / self.temperature) for task_loss in train_loss_list]

        return {"master": sum(balanced_loss), "det": balanced_loss[0], "seg": balanced_loss[1]}

    def update_meta_weights(self, train_image, train_target):
        if self.config["weight"] == "autol":
            val_image, val_target = self.meta_dataloader._next_data()

            val_target = [{k: v.to(self.device) for k, v in val_dict.items()} for val_dict in val_target]

            # format to [B,C,H,W] tensor
            if isinstance(val_image, tuple):
                val_image = plUtils.tuple_of_tensors_to_tensor(val_image).to(self.device)

            self.meta_optimizer.zero_grad()
            self.autol.unrolled_backward(
                train_image, train_target, val_image, val_target, self.lr_scheduler.get_last_lr()[0], self.optimizer
            )

            self.meta_optimizer.step()

    def loss_balancing_epoch_end(self, epoch, train_loss: dict):
        if self.config["weight"] in ["dynamic", "relative"]:

            train_loss_detached = {k: v.detach() for k, v in train_loss.items()}
            train_loss_detached.pop("master")

            if epoch == 1:
                self.nMinus2_loss = list(train_loss_detached.values())
            elif epoch == 2:
                self.nMinus1_loss = list(train_loss_detached.values())
            else:
                self.nMinus2_loss = self.nMinus1_loss
                self.nMinus1_loss = list(train_loss_detached.values())
