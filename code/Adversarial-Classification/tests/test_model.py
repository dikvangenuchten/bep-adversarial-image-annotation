import pytest
import torch

import torchvision
from model import MnistModel


@pytest.fixture(scope="module")
def sample_image():
    try:
        dataset = torchvision.datasets.mnist.MNIST(
            root="data/mnist",
            download=False,
            transform=torchvision.transforms.ToTensor(),
        )
    except RuntimeError:
        print("Could not find dataset locally, downloading")
        dataset = torchvision.datasets.mnist.MNIST(
            root="data/mnist",
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    return next(iter(data_loader))[0]


@pytest.fixture()
def mnist_model():
    return MnistModel()


def test_model_inference(mnist_model, sample_image):
    out = mnist_model.inference(sample_image)
    assert 0 <= out <= 9
