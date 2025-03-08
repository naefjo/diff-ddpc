from typing import List

import cvxpy as cp

from ..predictors.structures import Model
from .costs import CostFunction


class DataDrivenPredictiveController:
    def __init__(
        self, model: Model, cost: CostFunction, constraints: List[cp.Constraint]
    ):
        problem_constraints = model.constraints + constraints
        problem_cost = (
            cost.expr(model.variables["u"], model.variables["y"]) + model.regularizers
        )
        self._problem = cp.Problem(cp.Minimize(problem_cost), problem_constraints)

    def forward(self, args):
        pass

    def compute_action(self, args):
        pass
