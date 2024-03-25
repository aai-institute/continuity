import pytest
from continuity.operators.shape import TensorShape, OperatorShapes
from continuity.benchmarks.sine import SineBenchmark
from continuity.operators.integralkernel import NaiveIntegralKernel, NeuralNetworkKernel
from continuity.operators import NeuralOperator
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss
from .util import get_shape_mismatches


def test_shapes(random_shape_operator_datasets):
    latent_channels = 3
    operators = []
    for dataset in random_shape_operator_datasets:
        shapes = dataset.shapes
        hidden_shape = TensorShape(shapes.u.num, latent_channels)

        shapes = [
            OperatorShapes(x=shapes.x, u=hidden_shape, y=shapes.x, v=hidden_shape),
            OperatorShapes(x=shapes.x, u=hidden_shape, y=shapes.x, v=hidden_shape),
            OperatorShapes(
                x=shapes.x,
                u=hidden_shape,
                y=shapes.y,
                v=TensorShape(shapes.v.num, latent_channels),
            ),
        ]
        layers = [
            NaiveIntegralKernel(NeuralNetworkKernel(shapes[i], 32, 3))
            for i in range(len(shapes))
        ]
        operators.append(NeuralOperator(dataset.shapes, layers))

    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


@pytest.mark.slow
def test_neuraloperator():
    # Data set
    dataset = SineBenchmark(n_train=1).train_dataset
    shapes = dataset.shapes

    # Operator
    width, depth = 7, 9
    latent_shape = OperatorShapes(
        x=shapes.x,
        u=TensorShape(shapes.u.num, width),
        y=shapes.y,
        v=TensorShape(shapes.u.num, width),
    )
    layers = [
        NaiveIntegralKernel(NeuralNetworkKernel(latent_shape, 32, 3))
        for _ in range(depth)
    ]
    operator = NeuralOperator(dataset.shapes, layers)

    # Train
    Trainer(operator).fit(dataset, tol=1e-2)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-2
