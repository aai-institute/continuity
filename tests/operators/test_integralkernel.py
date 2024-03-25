import pytest
import torch

from continuity.benchmarks.sine import SineBenchmark
from continuity.operators.shape import OperatorShapes, TensorShape
from continuity.operators.integralkernel import (
    NeuralNetworkKernel,
    NaiveIntegralKernel,
    Kernel,
)
from .util import get_shape_mismatches


@pytest.fixture
def dirac_kernel():
    class Dirac(Kernel):
        def forward(self, x, y):
            x_reshaped, y_reshaped = x.unsqueeze(2), y.unsqueeze(1)
            dist = ((x_reshaped - y_reshaped) ** 2).sum(dim=-1)
            dist = dist.reshape(
                -1,
                self.shapes.x.num,
                self.shapes.y.num,
                self.shapes.u.dim,
                self.shapes.v.dim,
            )
            zero = torch.zeros(1)
            return torch.isclose(dist, zero).to(torch.get_default_dtype())

    return Dirac


def test_shapes(random_shape_operator_datasets):
    operators = []
    for dataset in random_shape_operator_datasets:
        kernel = NeuralNetworkKernel(
            shapes=dataset.shapes, kernel_depth=1, kernel_width=1
        )
        operators.append(NaiveIntegralKernel(kernel=kernel))
    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


def test_neuralnetworkkernel():
    n_obs = 8
    x_num, x_dim = 10, 2
    y_num, y_dim = 20, 3
    u_dim = 4
    v_dim = 1
    x = torch.rand(n_obs, x_num, x_dim)
    y = torch.rand(n_obs, y_num, y_dim)

    shapes = OperatorShapes(
        x=TensorShape(num=x_num, dim=x_dim),
        u=TensorShape(num=x_num, dim=u_dim),
        y=TensorShape(num=y_num, dim=y_dim),
        v=TensorShape(num=y_num, dim=v_dim),
    )

    # Kernel
    kernel = NeuralNetworkKernel(
        shapes=shapes,
        kernel_width=32,
        kernel_depth=1,
    )

    k = kernel(x, y)
    assert k.shape == (n_obs, x_num, y_num, u_dim, v_dim)


def test_naiveintegralkernel(dirac_kernel):
    # Data set
    dataset = SineBenchmark(n_train=1).train_dataset

    x, u, _, _ = [a.unsqueeze(0) for a in dataset[0]]

    # Kernel
    dirac = dirac_kernel(dataset.shapes)

    # Operator
    operator = NaiveIntegralKernel(kernel=dirac)

    # Create tensors
    y = torch.linspace(-1, 1, 32).reshape(1, -1, 1)

    # Apply operator
    v = operator(x.reshape((1, -1, 1)), u.reshape((1, -1, 1)), y.reshape((1, -1, 1)))

    # For num_sensors == num_evals, we get v = u / num_sensors.

    v_expected = u / 32
    assert (v == v_expected).all(), f"{v} != {v_expected}"
