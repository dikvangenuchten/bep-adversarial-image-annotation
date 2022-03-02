import torch
from model import MnistModel
from runner import Runner
import pytest


@pytest.fixture()
def zero_mnist_model():
    model = MnistModel()
    model = _set_model_weight_to_zero(model)
    model
    return model


def test_runner(zero_mnist_model):
    optimizer = torch.optim.SGD(zero_mnist_model.parameters(), lr=1, momentum=0.5)
    loss_func = torch.nn.functional.nll_loss

    test_runner = Runner(zero_mnist_model, optimizer, loss_func)
    x = torch.rand((1, 1, 28, 28))
    target = torch.tensor([1])

    # Ensure all outputs are equal
    pre_train_out = zero_mnist_model(x)
    assert len(torch.unique(pre_train_out)) == 1
    expected_loss = loss_func(zero_mnist_model(x), target)
    actual_loss = test_runner.train_step(x, target)
    assert expected_loss == actual_loss
    post_train_out = zero_mnist_model(x)
    assert not torch.allclose(pre_train_out, post_train_out)
    assert torch.argmax(post_train_out) == 1


def _set_model_weight_to_zero(model) -> torch.nn.Module:
    def _set_zero_weights_and_bias(m):
        if hasattr(m, "weight"):
            m.weight.data.fill_(0)
        if hasattr(m, "bias"):
            m.bias.data.fill_(0)

    return model.apply(_set_zero_weights_and_bias)
