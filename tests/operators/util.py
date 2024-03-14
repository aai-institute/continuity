from typing import List

from continuity.data import OperatorDataset
from continuity.operators import Operator


def eval_shapes_correct(
    operators: List[Operator], datasets: List[OperatorDataset]
) -> bool:
    """Evaluates if an operator outputs the same shape as a list of datasets expects.

    Args:
        operators: List of operator instances matching the list of datasets.
        datasets: list of operator datasets on which this property should be checked.

    Returns:
        True if the output of the operator is correct for all datasets, False otherwise.
    """
    for operator, dataset in zip(operators, datasets):
        x, u, y, v = dataset[:9]  # batch of size 9
        output = operator(x, u, y)
        if output.shape != v.shape:
            return False
    return True
