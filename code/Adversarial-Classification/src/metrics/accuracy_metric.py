from typing import Any, Dict
from metrics.abstract_metric import AbstractMetric
import torch


class AccuracyMetric(AbstractMetric):
    correct = 0
    total = 0

    def add_measurement(
        self: "AbstractMetric",
        input_: torch.Tensor,
        y_trues: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        predicted = torch.argmax(logits, dim=-1)
        self.correct += torch.sum(predicted == y_trues)
        self.total += y_trues.shape[0]

    def get_measurement(self) -> Dict[str, Any]:
        return {"accuracy": self.correct / self.total}
