import torch
from model import MnistModel
from runner import Runner
import pytest


@pytest.fixture()
def zero_mnist_model():
    model = MnistModel()
    model = _set_model_weight_to_zero(model)
    return model


@pytest.fixture()
def optimizer(zero_mnist_model):
    return torch.optim.SGD(zero_mnist_model.parameters(), lr=1, momentum=0.5)


@pytest.fixture()
def loss_func():
    return torch.nn.functional.nll_loss


@pytest.fixture()
def runner(zero_mnist_model, optimizer, loss_func):
    return Runner(zero_mnist_model, optimizer, loss_func)


def test_runner_train_step(zero_mnist_model, loss_func, runner):
    x = torch.rand((1, 1, 28, 28))
    target = torch.tensor([1])

    # Ensure all outputs are equal
    pre_train_out = zero_mnist_model(x)
    assert len(torch.unique(pre_train_out)) == 1
    expected_loss = loss_func(zero_mnist_model(x), target)

    actual_loss, actual_output = runner.train_step(x, target)

    assert torch.allclose(actual_output, pre_train_out)
    assert expected_loss == actual_loss
    post_train_out = zero_mnist_model(x)
    assert not torch.allclose(pre_train_out, post_train_out)
    assert torch.argmax(post_train_out) == 1


def test_runner_test_step(zero_mnist_model, runner):
    x = torch.rand((1, 1, 28, 28))
    target = torch.tensor([1])

    pre_train_out = zero_mnist_model(x)

    runner.test_step(x, target)

    post_train_out = zero_mnist_model(x)
    assert torch.allclose(pre_train_out, post_train_out)


def _set_model_weight_to_zero(model,) -> torch.nn.Module:
    def _set_zero_weights_and_bias(m):
        if hasattr(m, "weight"):
            m.weight.data.fill_(0)
        if hasattr(m, "bias"):
            m.bias.data.fill_(0)

    return model.apply(_set_zero_weights_and_bias)
