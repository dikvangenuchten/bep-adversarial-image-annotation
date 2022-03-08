"""Logs the input, target and output of the model in a table.
"""
from typing import Any, Dict
import torch
import abc


class AbstractMetric(abc.ABC):
    @abc.abstractmethod
    def add_measurement(
        self: "AbstractMetric",
        input_: torch.Tensor,
        y_trues: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Add a new measurement to the metric."""

    @abc.abstractmethod
    def get_measurement(self) -> Dict[str, Any]:
        """Retrieve a dictionary which can be logged to Wandb.

        Key is the name of the specific measurement.
        Value is the value of the specific measurement.
        """
