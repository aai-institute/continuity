import random
import torch
import mlflow
import numpy as np
from datetime import datetime
from typing import Optional
from continuity.benchmarks.run import RunConfig
from continuity.trainer import Trainer
from continuity.trainer.callbacks import MLFlowLogger
from continuity.data.utility import dataset_loss


class BenchmarkRunner:
    """Benchmark runner."""

    @staticmethod
    def run(config: RunConfig, params_dict: Optional[dict] = None) -> float:
        """Run a benchmark.

        Args:
            config: run configuration.
            params_dict: dictionary of parameters to log.

        Returns:
            Test loss.

        """
        # Benchmark
        benchmark = config.benchmark_factory()

        # Operator
        shapes = benchmark.train_dataset.shapes
        operator = config.operator_factory(shapes)

        # Print
        if params_dict is None:
            params_dict = {}

        param_str = " ".join(f"{key}={value}" for key, value in params_dict.items())
        print(f"> {benchmark} {operator} {param_str}")

        # MLFLow
        mlflow.set_experiment(f"{benchmark}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"{operator} {timestamp}"
        tags = {
            "benchmark": str(benchmark),
            "operator": str(operator),
            "device": str(config.device),
        }
        mlflow.start_run(run_name=run_name, tags=tags)

        # Log parameters
        if params_dict is not None:
            for key, value in params_dict.items():
                mlflow.log_param(key, value)

        if "seed" not in params_dict:
            mlflow.log_param("seed", config.seed)
        if "lr" not in params_dict:
            mlflow.log_param("lr", config.lr)
        if "batch_size" not in params_dict:
            mlflow.log_param("batch_size", config.batch_size)
        if "tol" not in params_dict:
            mlflow.log_param("tol", config.tol)
        if "max_epochs" not in params_dict:
            mlflow.log_param("max_epochs", config.max_epochs)
        mlflow.log_metric("num_params", operator.num_params())

        # Seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # For now, take the sum of all losses in benchmark
        def loss_fn(*args):
            return sum(loss(*args) for loss in benchmark.losses)

        # Trainer
        optimizer = torch.optim.Adam(operator.parameters(), lr=config.lr)
        trainer = Trainer(
            operator,
            optimizer,
            loss_fn=loss_fn,
            device=config.device,
            verbose=True,
        )

        # Train
        trainer.fit(
            benchmark.train_dataset,
            tol=config.tol,
            epochs=config.max_epochs,
            callbacks=[MLFlowLogger(operator)],
            batch_size=config.batch_size,
            test_dataset=benchmark.test_dataset,
        )

        # Return test loss
        return dataset_loss(benchmark.test_dataset, operator, loss_fn, config.device)
