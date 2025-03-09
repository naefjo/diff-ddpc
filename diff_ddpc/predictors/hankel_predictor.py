from typing import Tuple

import array_api_compat
import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from ..structures import DDPCDimensions, TrajectoryDataSet

from ..utils.logging import logger
from .structures import Model


def generate_hankel_matrices(
    data: TrajectoryDataSet, dimensions: DDPCDimensions
) -> Tuple[NDArray, NDArray]:
    """
    Generates the Hankel matrices for the Hankel matrix based predictor.
    """
    assert isinstance(data, TrajectoryDataSet)
    xp = array_api_compat.get_namespace(data.dataset[0].obs_trajectory)

    H_u_list: list[NDArray] = []
    H_y_list: list[NDArray] = []
    for trajectory in data.dataset:
        H_u_traj = block_hankel(
            trajectory.act_trajectory, dimensions.T_past + dimensions.T_fut
        )
        H_y_traj = block_hankel(
            trajectory.obs_trajectory, dimensions.T_past + dimensions.T_fut
        )
        H_u_list.append(H_u_traj)
        H_y_list.append(H_y_traj)

    H_u = xp.hstack(H_u_list)
    H_y = xp.hstack(H_y_list)
    return H_u, H_y


def _safe_cholesky(A, min_jitter: float = 1e-10, max_jitter: float = 1e-3):
    """
    Get Cholesky factor of A. Add jitter to the diagonal if A is not PD
    """
    xp = array_api_compat.get_namespace(A)
    jitter = min_jitter
    while True:
        if jitter > max_jitter:
            raise RuntimeError(
                f"Matrix is not PSD even after adding jitter up to {max_jitter}"
            )

        try:
            L = xp.linalg.cholesky(A @ A.T + jitter * xp.eye(A.shape[0]))
            break

        except np.linalg.LinAlgError:
            jitter *= 10

    logger.info(
        f"Added jitter of factor {jitter} to perform the low rank approximation.",
    )
    return L


def generate_hankel_predictor(
    data: TrajectoryDataSet,
    dimensions: DDPCDimensions,
    low_rank_approximation: bool = False,
) -> Model:
    """
    Generates a Hankel Matrix based predictor $Hg = [u; y]$.

    args:
        - data: the data set from which to generate the model
        - dimensions: the dimensions of the OCP
        - low_rank_approximation: Whether to perform an LQ factorization of
          the Hankel matrix (c.f. gamma-DDPC)
    returns:
        - Model: Hankel matrix based model
    """
    xp = array_api_compat.get_namespace(data.dataset[0].obs_trajectory)
    H_u, H_y = generate_hankel_matrices(data, dimensions)

    if low_rank_approximation:
        H = xp.vstack((H_u, H_y))
        L = _safe_cholesky(H)
        H_u, H_y = xp.vsplit(
            L, [(dimensions.T_past + dimensions.T_fut) * dimensions.n_act]
        )

    u_past = cp.Parameter((dimensions.T_past, dimensions.n_act))
    y_past = cp.Parameter((dimensions.T_past, dimensions.n_obs))
    u = cp.Variable((dimensions.T_fut, dimensions.n_act))
    y = cp.Variable((dimensions.T_fut, dimensions.n_obs))
    g = cp.Variable((H_u.shape[-1], 1))

    constraint = [
        H_u[: dimensions.T_past * dimensions.n_act, :] @ g == u_past.reshape((-1, 1)),
        H_u[dimensions.T_past * dimensions.n_act :, :] @ g == u.reshape((-1, 1)),
        H_y[: dimensions.T_past * dimensions.n_obs, :] @ g == y_past.reshape((-1, 1)),
        H_y[dimensions.T_past * dimensions.n_obs :, :] @ g == y.reshape((-1, 1)),
        cp.sum(g) == 1,
    ]
    u_mat = xp.vstack(
        (
            H_u[: dimensions.T_past * dimensions.n_act, :],
            H_u[dimensions.T_past * dimensions.n_act :, :],
            H_y[: dimensions.T_past * dimensions.n_obs, :],
            np.ones((1, H_u.shape[-1])),
        )
    )
    pi = xp.linalg.pinv(u_mat) @ u_mat

    return Model(
        constraint,
        {
            "u_past": u_past,
            "y_past": y_past,
            "u": u,
            "y": y,
        },
        {
            "g_1": cp.norm1(g),
            "g_pi": cp.sum_squares((xp.eye(pi.shape[0]) - pi) @ g),
        },
    )


def block_hankel(w: NDArray, L: int) -> NDArray:
    """
    Builds block Hankel matrix of order L from data w
    args:
        w : a T x d data matrix. T is the number of timesteps, d is the dimension of the signal
            e.g., if there are 6 timesteps and 4 entries at each timestep, w is 6 x 4
        L : order of hankel matrix
    """
    xp = array_api_compat.get_namespace(w)
    T = int(xp.shape(w)[0])  # number of timesteps
    d = int(xp.shape(w)[1])  # dimension of the signal
    if L > T:
        raise ValueError(f"L {L} must be smaller than T {T}")

    H = xp.zeros((L * d, T - L + 1))
    w_vec = w.reshape(-1)
    for i in range(0, T - L + 1):
        H[:, i] = w_vec[d * i : d * (L + i)]
    return H
