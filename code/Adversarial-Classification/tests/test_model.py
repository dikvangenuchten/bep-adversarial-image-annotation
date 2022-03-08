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


def test_model_load_save(mnist_model, tmpdir):
    model_path = str(tmpdir.join("test_model_load_save.pth"))
    x = torch.rand(1, 1, 28, 28)

    # Ensure model is in eval mode
    # dropout otherwise changes behaviour
    mnist_model.eval()
    pre_save_y = mnist_model(x)

    mnist_model.save(model_path)

    del mnist_model

    loaded_model = MnistModel.load(model_path)
    loaded_model.eval()
    load_y = loaded_model(x)
    assert torch.allclose(pre_save_y, load_y)
