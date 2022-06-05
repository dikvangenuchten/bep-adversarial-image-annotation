"""Runner trains and tests"""
from typing import Callable, List
import torch
from metrics import abstract_metric
import wandb

DEVICE = "cpu"


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.epoch: int = 0
        self.model: torch.nn.Module = model
        self.model.to(DEVICE)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = loss_func

    def train_epoch(
        self, dataset, metrics: List[abstract_metric.AbstractMetric]
    ) -> float:
        train_loss = []
        for x, target in dataset:
            loss, output = self.train_step(x, target)
            self._add_measurements_to_metrics(metrics, x, target, output)
            train_loss.append(loss)
        return sum(train_loss) / len(train_loss)

    @staticmethod
    def _add_measurements_to_metrics(metrics, x, target, output):
        for metric in metrics:
            metric.add_measurement(x, target, output)

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()

        x.to(DEVICE)
        output = self.model(x)
        loss = self.loss_func(output, target.to(DEVICE))

        loss.backward()
        self.optimizer.step()
        return loss.cpu(), output.cpu()

    def test_step(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_func(output, target)

        return loss, output
