"""Runner trains and tests"""
from typing import Callable
import torch


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.epoch: int = 0
        self.model: torch.nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_func: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = loss_func

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

    def test_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_func(output, target)

        return loss
