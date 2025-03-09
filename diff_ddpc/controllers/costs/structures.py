from dataclasses import dataclass
from typing import Callable, Dict

import cvxpy as cp


@dataclass
class CostFunction:
    """
    Structure to represent a cost function and the corresponding parameters.
    """

    expr: Callable[[cp.Expression, cp.Expression], cp.Expression]
    params: Dict[str, cp.Parameter]
