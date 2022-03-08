"""Runner trains and tests"""
from typing import Callable, List
import torch
from metrics import abstract_metric


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

    def train_epoch(
        self,
        dataset,
        epoch,
        metrics: List[abstract_metric.AbstractMetric],
    ):
        train_loss = []
        for x, target in dataset:
            loss, output = self.train_step(x, target)
            for metric in metrics:
                metric.add_measurement(
                    x,
                    target,
                    output,
                )
            train_loss.append(loss)
        wandb.log(
            {"epoch": epoch, "train-loss": sum(train_loss) / len(train_loss)}
        )

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
        return loss, output

    def test_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_func(output, target)

        return loss, output
