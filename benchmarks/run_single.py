from continuity.benchmarks.run import BenchmarkRunner, RunConfig
from continuity.benchmarks import SineRegular
from continuity.operators import DeepNeuralOperator

config = RunConfig(
    benchmark_factory=SineRegular,
    operator_factory=DeepNeuralOperator,
    seed=0,
    lr=1e-3,
    tol=0,
    max_epochs=100,
    batch_size=32,
    device="cpu",
)

if __name__ == "__main__":
    BenchmarkRunner.run(config)
