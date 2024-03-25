import pytest
import torch
from itertools import product
from typing import List
from continuity.data import OperatorDataset


@pytest.fixture(scope="session")
def random_shape_operator_datasets() -> List[OperatorDataset]:
    """

    Returns:
        List containing permutations of 1D and 2D vectors (x, u, y, v) in OperatorDataset.
    """
    n_sensors = 5
    n_evaluations = 7
    n_observations = 11
    x_dims = (1, 2)
    u_dims = (1, 2)
    y_dims = (1, 2)
    v_dims = (1, 2)

    datasets = []

    for x_dim, u_dim, y_dim, v_dim in product(x_dims, u_dims, y_dims, v_dims):
        x_samples = torch.rand(n_observations, n_sensors, x_dim)
        u_samples = torch.rand(n_observations, n_sensors, u_dim)
        y_samples = torch.rand(n_observations, n_evaluations, y_dim)
        v_samples = torch.rand(n_observations, n_evaluations, v_dim)
        datasets.append(
            OperatorDataset(x=x_samples, u=u_samples, y=y_samples, v=v_samples)
        )

    return datasets
