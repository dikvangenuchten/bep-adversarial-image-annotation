import pytest
from metrics.accuracy_metric import AccuracyMetric
import torch


@pytest.fixture(params=[1, 2, 16])
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 16])
def n_classes(request) -> int:
    return request.param


@pytest.fixture()
def accuracy_metric():
    return AccuracyMetric()


def test_all_correct(batch_size, n_classes, accuracy_metric):
    x = torch.rand((batch_size, 8, 8))
    predicted_y = torch.cat(
        [torch.ones((batch_size, 1)), torch.zeros((batch_size, n_classes - 1))],
        dim=-1,
    )
    target_y = torch.tensor([0] * batch_size)

    accuracy_metric.add_measurement(x, target_y, predicted_y)

    accuracy_dict = accuracy_metric.get_measurement()
    assert len(accuracy_dict) == 1
    accuracy = accuracy_dict["accuracy"]
    assert accuracy == 1


def test_all_incorrect(batch_size, n_classes, accuracy_metric):
    x = torch.rand((batch_size, 8, 8))
    predicted_y = torch.cat(
        [torch.ones((batch_size, 1)), torch.zeros((batch_size, n_classes - 1))],
        dim=-1,
    )
    target_y = torch.tensor([1] * batch_size)

    accuracy_metric.add_measurement(x, target_y, predicted_y)

    accuracy_dict = accuracy_metric.get_measurement()
    assert len(accuracy_dict) == 1
    accuracy = accuracy_dict["accuracy"]
    assert accuracy == 0

def test_2_batches(batch_size, n_classes, accuracy_metric):
    x = torch.rand((batch_size, 8, 8))
    predicted_y = torch.cat(
        [torch.ones((batch_size, 1)), torch.zeros((batch_size, n_classes - 1))],
        dim=-1,
    )
    # These should be incorrect
    target_y = torch.tensor([1] * batch_size)
    accuracy_metric.add_measurement(x, target_y, predicted_y)
    
    # These should be correct
    target_y = torch.tensor([0] * batch_size)
    accuracy_metric.add_measurement(x, target_y, predicted_y)
    

    accuracy_dict = accuracy_metric.get_measurement()
    assert len(accuracy_dict) == 1
    accuracy = accuracy_dict["accuracy"]
    assert accuracy == 0.5
