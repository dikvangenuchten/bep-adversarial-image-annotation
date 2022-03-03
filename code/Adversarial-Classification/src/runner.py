"""Runner trains and tests"""
from typing import Callable
import torch


class Runner:
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
    ) -> None:
        self.epoch = 0
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

    def train_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        self.optimizer.zero_grad()

        output = self.model(x)
        loss = self.loss_func(output, target)

        loss.backward()
        self.optimizer.step()
        return loss
