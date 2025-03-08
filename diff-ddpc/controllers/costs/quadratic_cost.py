from typing import Union

import cvxpy as cp
import numpy as np
import torch
from numpy.typing import NDArray

from ... import DDPCDimensions

ArrayLike = Union[NDArray, torch.Tensor]


def get_quadratic_cost(
    Q: ArrayLike, R: ArrayLike, u: cp.Variable, y: cp.Variable, dims: DDPCDimensions
) -> cp.Expression:
    """
    Simple quadratic cost function.
    TODO(@naefjo): Fix type hint for u and y so basedpyright shuts up...
    """
    return cp.quad_form(y.reshape(-1), cp.kron(np.eye(dims.T_fut), Q)) + cp.quad_form(
        u.reshape(-1), cp.kron(np.eye(dims.T_fut), R)
    )


def get_quadratic_tracking_cost(
    Q: ArrayLike,
    R: ArrayLike,
    u: cp.Variable,
    y: cp.Variable,
    u_ref: ArrayLike,
    y_ref: ArrayLike,
    dims: DDPCDimensions,
) -> cp.Expression:
    """
    Quadratic tracking cost function.
    """
    return get_quadratic_cost(Q, R, u - u_ref, y - y_ref, dims)
