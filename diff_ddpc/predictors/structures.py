from dataclasses import dataclass
from typing import Dict, List

import cvxpy as cp


@dataclass
class Model:
    constraints: List[cp.Constraint]
    variables: Dict[str, cp.Variable | cp.Parameter]
    regularizers: Dict[str, cp.Expression]
