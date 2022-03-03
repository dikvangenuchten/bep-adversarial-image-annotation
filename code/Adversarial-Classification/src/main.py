from statistics import mean
from typing import Callable
import torch
import torchvision
from model import MnistModel
from runner import Runner

LEARNING_RATE = 0.01
MOMENTUM = 0.5
BATCH_SIZE = 64


def get_mnist() -> torch.utils.data.Dataset:
    return torch.utils.data.DataLoader(
        torchvision.datasets.mnist.MNIST(
            root="data/mnist",
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


if __name__ == "__main__":
    dataset = get_mnist()
    model = MnistModel()
    model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )
    loss_func = torch.nn.functional.nll_loss
    runner = Runner(model, optimizer, loss_func)

    for x, target in dataset:
        x = x.cuda()
        target = target.cuda()
        runner.train_step(x, target)
