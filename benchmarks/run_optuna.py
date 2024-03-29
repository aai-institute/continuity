import optuna
from functools import partial
from continuity.benchmarks.run import BenchmarkRunner, RunConfig
from continuity.benchmarks import SineRegular
from continuity.operators import (
    FourierNeuralOperator,
)


if __name__ == "__main__":

    def objective(trial):
        seed = trial.suggest_int("seed", 0, 100)
        width = trial.suggest_int("width", 1, 4)
        depth = trial.suggest_int("depth", 1, 4)

        config = RunConfig(
            SineRegular,
            partial(FourierNeuralOperator, width=width, depth=depth),
            max_epochs=100,
            seed=seed,
        )

        test_loss = BenchmarkRunner.run(config, trial.params)
        return test_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
