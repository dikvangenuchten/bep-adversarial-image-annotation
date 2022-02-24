import torch
import torchvision


def get_mnist() -> torch.utils.data.Dataset:
    return torchvision.datasets.mnist.MNIST(root="data/mnist", download=True)


if __name__ == "__main__":
    dataset = get_mnist()
    for (img, label) in dataset:
        print(label)
