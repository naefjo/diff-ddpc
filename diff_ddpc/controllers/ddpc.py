from typing import Dict, List
from warnings import warn

import array_api_compat
import cvxpy as cp

from diff_ddpc.structures import DDPCDimensions

from ..predictors.structures import Model
from .base_controller import BaseController
from .costs import CostFunction


class DataDrivenPredictiveController(BaseController):
    """
    Implements a (fixed) Predictive Controller using CVXPY.

    Here, fixed indicates that the predictive 'model' does not change throughout the QP solves.
    """

    def __init__(
        self,
        dims: DDPCDimensions,
        model: Model,
        cost: CostFunction,
        constraints: List[cp.Constraint],
        solver_opts: Dict,
    ):
        """
        args:
            - dims: Dimensions of the control problem
            - model: The predictive model used in the controller
            - cost: CostFunction instance describing the cost to be *minimized*
            - constraints: Additional constraints to consider in the optimal control problem
            - solver_opts: Arguments which are passed to the CVXPY solver.
        """
        self._dims = dims
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

        self._initialized = False

    def forward(self, obs, *args, **kwargs):
        if not self._initialized:
            self._initialize(obs)
            self._initialized = True

        if self.set_params != self._params.keys():
            warn("Not all parameters set in the optimization problem")

        self._set_initial_obs(obs)
        # TODO: Deal with solver failure.
        self._problem.solve(**self._solver_opts)
        self.set_params = set()

        action = self._model.variables["u"].value[0, :]

        self._set_initial_action(action)

        return action

    def _initialize(self, obs):
        xp = array_api_compat.array_namespace(obs)
        self._model.variables["y_past"] = xp.tile(obs, (self._dims.T_fut, 1))

    def _set_initial_obs(self, obs) -> None:
        """
        Update the cyclic buffer y_past with obs
        """
        self._model.variables["y_past"].value[:-1, :] = self._model.variables[
            "y_past"
        ].value[1:, :]
        self._model.variables["y_past"].value[-1, :] = obs

    def _set_initial_action(self, action) -> None:
        """
        Update the cyclic buffer u_past with obs
        """
        self._model.variables["u_past"].value[:-1, :] = self._model.variables[
            "u_past"
        ].value[1:, :]
        self._model.variables["u_past"].value[-1, :] = action

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
