"""Logs the input, target and output of the model in a table.
"""
from typing import Any, Dict, List
import torch
import wandb
from metrics.abstract_metric import AbstractMetric


class WandbTableMetric(AbstractMetric):
    def __init__(self, labels: List[str]) -> None:
        self._labels = labels
        self._id = 0
        self._table = wandb.Table(
            columns=[
                "id",
                "image",
                "y_true",
                *[f"logit: {label}" for label in self._labels],
            ]
        )

    def add_measurement(
        self: "WandbTableMetric",
        input_: torch.Tensor,
        y_trues: torch.Tensor,
        logits: torch.Tensor,
    ):
        for x, y_true, logit in zip(input_, y_trues, logits):
            self._table.add_data(
                self._id, wandb.Image(x), y_true, *logit.split(1, dim=-1)
            )
            self._id += 1

    def get_measurement(self) -> Dict[str, Any]:
        return {"Sample Table": self._table}
