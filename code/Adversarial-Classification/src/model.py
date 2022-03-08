from typing import Union
from torch import nn
import torch


class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(),
        )

    @classmethod
    def load(cls, path: Union[str, dict]) -> "MnistModel":
        if isinstance(path, str):
            path = torch.load(path)
        model = cls()
        model.load_state_dict(path)
        return model

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, x):
        return self.model(x)

    def inference(self, x):
        out = self.forward(x)
        return torch.argmax(out, -1)
