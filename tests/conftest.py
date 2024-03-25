import pytest
import random
import torch
import numpy as np

pytest_plugins = [
    "tests.transforms.fixtures",
    "tests.operators.fixtures",
]


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
