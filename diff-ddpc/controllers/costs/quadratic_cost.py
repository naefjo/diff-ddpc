from typing import Union

import cvxpy as cp
import numpy as np
import torch
from numpy.typing import NDArray

from ... import DDPCDimensions
from . import CostFunction

ArrayLike = Union[NDArray, torch.Tensor, cp.Parameter]


def get_quadratic_cost(
    Q: ArrayLike, R: ArrayLike, dims: DDPCDimensions
) -> CostFunction:
    """
    Simple quadratic cost function.
    TODO(@naefjo): Fix type hints so basedpyright shuts up...
    """
    if Q.shape[0] == dims.n_obs and R.shape[0] == dims.n_act:
        Q_mat = cp.kron(np.eye(dims.T_fut), Q)
        R_mat = cp.kron(np.eye(dims.T_fut), R)
    elif (
        Q.shape[0] == dims.n_obs * dims.T_fut and R.shape[0] == dims.T_fut * dims.n_act
    ):
        Q_mat = Q
        R_mat = R
    else:
        raise RuntimeError("Shapes of requested Q and R not supported")

    return CostFunction(
        lambda u, y: cp.quad_form(y.reshape(-1), Q_mat)
        + cp.quad_form(u.reshape(-1), R_mat),
        dict(),
    )


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
    return CostFunction(
        lambda u, y: quadratic_cost.expr(u - u_ref, y - y_ref),
        {
            "u_ref": u_ref,
            "y_ref": y_ref,
        },
    )


def get_parametrized_quadratic_tracking_cost(dims: DDPCDimensions):
    """
    Generic parametrized quadratic cost function where the weight matrices
    are treated as (possibly varying) parameters

    TODO(@naefjo): This is likely busted. Check impl of Q in `get_quadratic_cost`
    """
    Q = cp.Parameter(dims.n_obs * dims.T_fut)
    R = cp.Parameter(dims.n_act * dims.T_fut)

    quadratic_tracking_cost = get_quadratic_tracking_cost(Q, R, dims)

    quadratic_tracking_cost.params.update({"Q": Q, "R": R})
    return quadratic_tracking_cost
