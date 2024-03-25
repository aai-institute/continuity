import pytest
from typing import List
from continuity.benchmarks.sine import SineBenchmark
from continuity.trainer import Trainer
from continuity.operators import DeepNeuralOperator
from continuity.operators.losses import MSELoss

from .util import get_shape_mismatches


@pytest.fixture(scope="module")
def dnos(random_shape_operator_datasets) -> List[DeepNeuralOperator]:
    return [
        DeepNeuralOperator(dataset.shapes) for dataset in random_shape_operator_datasets
    ]


def test_can_initialize(dnos):
    for operator in dnos:
        assert isinstance(operator, DeepNeuralOperator)


def test_shapes(dnos, random_shape_operator_datasets):
    assert get_shape_mismatches(dnos, random_shape_operator_datasets) == []


@pytest.mark.slow
def test_does_converge():
    # Data set
    benchmark = SineBenchmark(n_train=1)
    dataset = benchmark.train_dataset

    # Operator
    operator = DeepNeuralOperator(dataset.shapes)

    # Train
    Trainer(operator, lr=1e-2).fit(dataset, tol=1e-3)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-3
