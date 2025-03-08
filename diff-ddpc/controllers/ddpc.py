from typing import Dict, List
from warnings import warn

import cvxpy as cp

from ..predictors.structures import Model
from .base_controller import BaseController
from .costs import CostFunction


class DataDrivenPredictiveController(BaseController):
    def __init__(
        self,
        model: Model,
        cost: CostFunction,
        constraints: List[cp.Constraint],
        solver_opts: Dict,
    ):
        self._model = model
        self._cost = cost
        self._constraints = constraints
        self._solver_opts = solver_opts

        problem_constraints = model.constraints + constraints
        problem_cost = (
            cost.expr(model.variables["u"], model.variables["y"]) + model.regularizers
        )
        self._problem = cp.Problem(cp.Minimize(problem_cost), problem_constraints)

        self._params: dict[str, cp.Parameter] = dict()
        self._params.update(cost.params)
        self.set_params = set()

    def forward(self, argsstate, reference, *args, **kwargs):
        if self.set_params != self._params.keys():
            warn("Not all parameters set in the optimization problem")

        self._problem.solve(**self._solver_opts)
        self.set_params = set()

        self._update_internal_state()

        return self._model.variables["u"][0, :]

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

    def _update_internal_state(self):
        """
        Updates the past states u_p and y_p
        """
        raise NotImplementedError
