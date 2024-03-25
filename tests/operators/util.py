from typing import List

from continuity.data import OperatorDataset
from continuity.operators.shape import OperatorShapes
from continuity.operators import Operator


def get_shape_mismatches(
    operators: List[Operator], datasets: List[OperatorDataset]
) -> List[OperatorShapes]:
    """Evaluates if an operator outputs the same shape as a list of datasets expects.

    Args:
        operators: List of operator instances matching the list of datasets.
        datasets: list of operator datasets on which this property should be checked.

    Returns:
        List of `DatasetShapes` for which the evaluation failed.
    """
    failed_shapes = []
    for operator, dataset in zip(operators, datasets):
        x, u, y, v = dataset[:9]  # batch of size 9
        try:
            output = operator(x, u, y)
            if output.shape != v.shape:
                failed_shapes.append(dataset.shapes)
        except AssertionError or RuntimeError:
            failed_shapes.append(dataset.shapes)
    return failed_shapes
