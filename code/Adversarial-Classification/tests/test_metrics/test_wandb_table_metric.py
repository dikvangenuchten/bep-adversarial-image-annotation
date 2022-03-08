from metrics.wandb_table_metric import WandbTableMetric

import pytest
import torch
import numpy as np


@pytest.fixture(params=[1, 2, 16])
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 16])
def n_classes(request) -> int:
    return request.param


@pytest.fixture(scope="function")
def wandb_table_metric(n_classes):
    return WandbTableMetric(list(range(n_classes)))


def test_single_step(wandb_table_metric, n_classes, batch_size):
    x = torch.rand((batch_size, 8, 8))
    predicted_y = torch.rand((batch_size, n_classes))
    target_y = [1] * batch_size

    wandb_table_metric.add_measurement(x, target_y, predicted_y)

    table = wandb_table_metric.get_measurement()["Sample Table"]

    assert batch_size == len(table.data), "Not all samples have been logged."
    # Id, Image and target + 1 per logit
    assert 3 + n_classes == len(table.data[0]), "Not all rows have been logged."
    assert batch_size == len(
        np.unique(list(row[0] for row in table.data))
    ), "Not all ids are unique"
