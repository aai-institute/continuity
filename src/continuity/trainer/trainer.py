"""
`continuity.trainer.trainer`
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, List, Union
from continuity.data import OperatorDataset, dataset_loss
from continuity.operators import Operator
from continuity.operators.losses import Loss, MSELoss
from continuity.trainer.device import get_device
from .callbacks import Callback, PrintTrainingLoss
from .scheduler import LinearLRScheduler
from .criterion import Criterion, TrainingLossCriterion, TestLossCriterion
from .logs import Logs


class Trainer:
    """Trainer implements a default training loop for operator learning.

    Example:
        ```python
        from continuity.trainer import Trainer
        from continuity.operators.losses import MSELoss
        ...
        optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
        loss_fn = MSELoss()
        trainer = Trainer(operator, optimizer, loss_fn, device="cuda:0")
        trainer.fit(dataset, tol=1e-3, epochs=1000)
        ```

    Args:
        operator: Operator to be trained.
        optimizer: Torch-like optimizer. Default is Adam with learning rate `lr`.
        lr: Learning rate. Ignored if optimizer is not None. Default is 1e-3.
        loss_fn: Loss function taking (op, x, u, y, v). Default is MSELoss.
        device: Device to train on. Default is CPU.
        verbose: Print model parameters and use PrintTrainingLoss callback by default. Default is True.
    """

    device = get_device()

    def __init__(
        self,
        operator: Operator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        loss_fn: Optional[Loss] = None,
        device: torch.device = device,
        verbose: Optional[bool] = None,
    ):
        self.operator = operator
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(operator.parameters(), lr=lr)
        )
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.device = device

        # Verbosity
        if self.device.index is not None:
            self.verbose = verbose or self.device.index == 0
        else:
            self.verbose = verbose or True

    def fit(
        self,
        dataset: OperatorDataset,
        tol: float = 1e-5,
        epochs: int = 1000,
        callbacks: Optional[List[Callback]] = None,
        criterion: Optional[Criterion] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        test_dataset: Optional[OperatorDataset] = None,
        lr_scheduler: Union[bool, Callback] = True,
    ):
        """Fit operator to data set.

        Args:
            dataset: Data set.
            tol: Tolerance for stopping criterion. Ignored if criterion is not None.
            epochs: Maximum number of epochs.
            callbacks: List of additional callbacks.
            criterion: Stopping criterion. Defaults to TrainingLossCriteria(tol).
            batch_size: Batch size.
            shuffle: Shuffle data set.
            test_dataset: Test data set.
            lr_scheduler: Learning rate scheduler. If True, `LinearLRScheduler` is used.
        """
        # Callbacks
        callbacks = callbacks or []

        if self.verbose:
            callbacks.append(PrintTrainingLoss())

        if lr_scheduler is not False:
            if lr_scheduler is True:
                lr_scheduler = LinearLRScheduler(self.optimizer, epochs)
            callbacks.append(lr_scheduler)

        # Default criterion
        if criterion is None:
            if test_dataset is None:
                criterion = TrainingLossCriterion(tol)
            else:
                criterion = TestLossCriterion(tol)

        # Print number of model parameters
        if self.verbose:
            num_params = self.operator.num_params()
            print(f"Model parameters: {num_params}")

        # Move operator to device
        operator = self.operator.to(self.device)

        # Use DistributedDataParallel if available
        sampler = None
        if dist.is_available() and dist.is_initialized():
            operator = DDP(
                operator, device_ids=[self.device], output_device=self.device
            )
            sampler = DistributedSampler(dataset)
            shuffle = False

            if self.verbose:
                ngpu = dist.get_world_size()
                print(f"Device: CUDA ({ngpu} GPU{'' if ngpu == 1 else 's'})")
        else:
            if self.verbose:
                print(f"Device: {self.device}")

        # Create data loader
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
        )

        # Call on_train_begin
        for callback in callbacks:
            callback.on_train_begin()

        # Train
        loss_train, loss_test, epoch = None, None, 0
        operator.train()
        for epoch in range(epochs):
            loss_train = 0

            for x, u, y, v in data_loader:
                x, u = x.to(self.device), u.to(self.device)
                y, v = y.to(self.device), v.to(self.device)

                def closure(x=x, u=u, y=y, v=v):
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(operator, x, u, y, v)
                    loss.backward(retain_graph=True)
                    return loss

                loss = self.optimizer.step(closure)

                # Compute mean loss
                loss_train += loss.detach().item()

            loss_train /= len(data_loader)

            # Compute test loss
            if test_dataset is not None:
                loss_test = dataset_loss(
                    test_dataset, operator, self.loss_fn, self.device
                )

            # Callbacks
            logs = Logs(
                epoch=epoch + 1,
                loss_train=loss_train,
                loss_test=loss_test,
            )

            for callback in callbacks:
                callback(logs)

            # Stopping criterion
            if criterion is not None:
                if criterion(logs):
                    break

        # Call on_train_end
        for callback in callbacks:
            callback.on_train_end()

        # Move operator back to CPU
        self.operator.to("cpu")
