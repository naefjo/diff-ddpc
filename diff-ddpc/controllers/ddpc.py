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

        self._params: dict[str, cp.Parameter] = dict()
        self._params.update(cost.params)
        self.set_params = set()

    def forward(self, args):
        pass

    def compute_action(self, args):
        pass

    def get_parameter_list(self):
        return self._params.keys()

    def update_param(self, name: str, val) -> None:
        """
        Update the value of a registered parameter
        """
        assert (
            name in self._params
        ), f"Parameter {name} not in the registered parameters."

        self._params[name].value = val
        self.set_params.add(name)
