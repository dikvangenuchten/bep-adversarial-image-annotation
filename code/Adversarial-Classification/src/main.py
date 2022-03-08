import torch
import torchvision
from model import MnistModel
from runner import Runner
from metrics import wandb_table_metric
from metrics.abstract_metric import AbstractMetric
from metrics.accuracy_metric import AccuracyMetric
import wandb
from typing import List

LEARNING_RATE = 0.01
MOMENTUM = 0.5
BATCH_SIZE = 64


def get_mnist(train: bool) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        torchvision.datasets.mnist.MNIST(
            root="data/mnist",
            download=True,
            transform=torchvision.transforms.ToTensor(),
            train=train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


def main():
    train_dataset = get_mnist(train=True)
    test_dataset = get_mnist(train=False)
    model = MnistModel()
    wandb.watch(model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )
    loss_func = torch.nn.functional.nll_loss

    # Used for both train and test
    metric_factories = [AccuracyMetric]
    # Used only for test
    test_metric_factories = [
        # lambda: wandb_table_metric.WandbTableMetric(list(range(10)))
    ]

    train_model(
        train_dataset,
        test_dataset,
        model,
        optimizer,
        loss_func,
        metric_factories,
        test_metric_factories,
    )

    torch.save(model.state_dict(), "data/model.pth")


def train_model(
    train_dataset,
    test_dataset,
    model,
    optimizer,
    loss_func,
    metric_factories,
    test_metric_factories,
):
    runner = Runner(model, optimizer, loss_func)
    for epoch in range(15):
        train_metrics = [
            metric_factory() for metric_factory in metric_factories
        ]
        train_loss = runner.train_epoch(train_dataset, train_metrics)

        test_metrics = [metric_factory() for metric_factory in metric_factories]
        test_metrics.extend(
            metric_factory() for metric_factory in test_metric_factories
        )
        test_epoch(test_dataset, runner, epoch, test_metrics)
        log_metrics(train_metrics, test_metrics, epoch)
        print(f"Finished epoch: {epoch}")


def test_model(model, dataset, metrics, loss_func):
    """Run a single epoch logging to wandb"""
    test_loss = []
    for x, target in dataset:
        output = model(x)
        loss = loss_func(output, target)
        for metric in metrics:
            metric.add_measurement(
                x,
                target,
                output,
            )
        test_loss.append(loss)


def log_metrics(train_metrics, test_metrics, epoch):
    wandb.log(
        {
            "train": {
                k: v
                for metric in train_metrics
                for k, v in metric.get_measurement().items()
            },
            "test": {
                k: v
                for metric in test_metrics
                for k, v in metric.get_measurement().items()
            },
        },
        step=epoch,
        commit=True,
    )


def test_epoch(
    dataset,
    runner: Runner,
    epoch,
    metrics: List[wandb_table_metric.WandbTableMetric],
):
    test_loss = []
    for x, target in dataset:
        loss, output = runner.train_step(x, target)
        for metric in metrics:
            metric.add_measurement(
                x,
                target,
                output,
            )
        test_loss.append(loss)

    metrics_data = {
        "Test epoch": epoch,
        "Test loss": sum(test_loss) / len(test_loss),
    }
    for metric in metrics:
        # Rename metric
        metric_data = {
            f"Test {key}": value
            for key, value in metric.get_measurement().items()
        }
        metrics_data.update(metric_data)


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project="Adversarial Mnist",
        settings=wandb.Settings(start_method="fork"),
    )
    main()
