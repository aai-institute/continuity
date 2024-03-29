"""
`continuity.operators.losses`

Loss functions for operator learning.

Every loss function takes an operator `op`, sensor positions `x`, sensor values `u`,
evaluation coordinates `y`, and labels `v` as input and returns a scalar loss:

```python
loss = loss_fn(op, x, u, y, v)
```
"""

import torch
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continuity.operators.operator import Operator


class Loss:
    """Loss function for training operators in Continuity."""

    @abstractmethod
    def __call__(
        self,
        op: "Operator",
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate loss.

        Args:
            op: Operator object
            x: Tensor of sensor positions of shape (batch_size, num_sensors, coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, num_channels)
            y: Tensor of evaluation coordinates of shape (batch_size, x_size, coordinate_dim)
            v: Tensor of labels of shape (batch_size, x_size, coordinate_dim)
        """


class MSELoss(Loss):
    """Computes the mean-squared error between the predicted and true labels.

    ```python
    loss = mse(op(x, u, y), v)
    ```
    """

    def __init__(self):
        self.mse = torch.nn.MSELoss()

    def __call__(
        self,
        op: "Operator",
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate MSE loss.

        Args:
            op: Operator object
            x: Tensor of sensor positions of shape (batch_size, num_sensors, coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, num_channels)
            y: Tensor of evaluation coordinates of shape (batch_size, x_size, coordinate_dim)
            v: Tensor of labels of shape (batch_size, x_size, coordinate_dim)
        """
        # Call operator
        v_pred = op(x, u, y)

        # Align shapes
        v_pred = v_pred.reshape(v.shape)

        # Return MSE
        return self.mse(v_pred, v)
