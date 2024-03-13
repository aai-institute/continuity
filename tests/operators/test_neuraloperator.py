import pytest
from continuity.operators.shape import TensorShape, OperatorShapes
from continuity.benchmarks.sine import SineBenchmark
from continuity.operators.integralkernel import NaiveIntegralKernel, NeuralNetworkKernel
from continuity.operators import NeuralOperator
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss
from .util import eval_shapes_correct


def test_shapes(random_shape_operator_datasets):
    assert eval_shapes_correct(NeuralOperator, random_shape_operator_datasets)


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
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3


if __name__ == "__main__":
    test_neuraloperator()
