from typing import Union

import cvxpy as cp
import numpy as np
import torch
from numpy.typing import NDArray

from ... import DDPCDimensions

ArrayLike = Union[NDArray, torch.Tensor]


def get_quadratic_cost(Q: ArrayLike, R: ArrayLike, dims: DDPCDimensions):
    """
    Simple quadratic cost function.
    TODO(@naefjo): Fix type hints so basedpyright shuts up...
    """
    return {
        "cost_fun": lambda u, y: cp.quad_form(
            y.reshape(-1), cp.kron(np.eye(dims.T_fut), Q)
        )
        + cp.quad_form(u.reshape(-1), cp.kron(np.eye(dims.T_fut), R)),
        "params": dict(),
    }


def get_quadratic_tracking_cost(
    Q: ArrayLike,
    R: ArrayLike,
    dims: DDPCDimensions,
):
    """
    Quadratic tracking cost function.
    """
    quadratic_cost = get_quadratic_cost(Q, R, dims)

    u_ref = cp.Parameter((dims.T_fut, dims.n_act))
    y_ref = cp.Parameter((dims.T_fut, dims.n_obs))
    return {
        "cost_fun": lambda u, y: quadratic_cost["cost_fun"](u - u_ref, y - y_ref),
        "params": {
            "u_ref": u_ref,
            "y_ref": y_ref,
        },
    }
