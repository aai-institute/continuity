from typing import List

from continuity.data import OperatorDataset
from continuity.operators import Operator


def eval_shapes_correct(
    operator_class: type(Operator), datasets: List[OperatorDataset], *args, **kwargs
) -> bool:
    """Evaluates if an operator outputs the same shape as a list of datasets expects.

    Args:
        operator_class: Operator class.
        datasets: list of operator datasets on which this property should be checked.
        *args: List of additional arguments for the instantiation of the operator.
        **kwargs: Dict of additional arguments for the instantiation of the operator.

    Returns:
        True if the output of the operator is correct for all datasets, False otherwise.
    """
    for dataset in datasets:
        operator = operator_class(dataset.shapes, *args, **kwargs)
        x, u, y, v = dataset[:9]  # batch of size 9
        output = operator(x, u, y)
        if output.shape != v.shape:
            return False

    return True
