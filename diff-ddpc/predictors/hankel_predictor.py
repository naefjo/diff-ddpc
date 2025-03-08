from typing import Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .. import DDPCDimensions, TrajectoryDataSet
from .structures import Model


def generate_hankel_matrices(
    data: TrajectoryDataSet, dimensions: DDPCDimensions
) -> Tuple[NDArray, NDArray]:
    """
    Generates the Hankel matrices for the Hankel matrix based predictor.
    """
    assert isinstance(data, TrajectoryDataSet)

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

    H_u = np.hstack(H_u_list)
    H_y = np.hstack(H_y_list)
    return H_u, H_y


def generate_hankel_predictor(
    data: TrajectoryDataSet,
    dimensions: DDPCDimensions,
) -> Model:
    """
    Generates a Hankel Matrix based predictor Hg = [u; y].
    """
    H_u, H_y = generate_hankel_matrices(data, dimensions)
    u_past = cp.Variable((dimensions.T_past, dimensions.n_act))
    y_past = cp.Variable((dimensions.T_past, dimensions.n_obs))
    u = cp.Variable((dimensions.T_fut, dimensions.n_act))
    y = cp.Variable((dimensions.T_fut, dimensions.n_obs))
    g = cp.Variable(H_u.shape[-1])

    constraint = [
        H_u[: dimensions.T_past * dimensions.n_act, :] @ g == u_past.reshape((-1, 1)),
        H_u[dimensions.T_past * dimensions.n_act :, :] @ g == u.reshape((-1, 1)),
        H_y[: dimensions.T_past * dimensions.n_obs, :] @ g == y_past.reshape((-1, 1)),
        H_y[dimensions.T_past * dimensions.n_obs :, :] @ g == y.reshape((-1, 1)),
        cp.sum(g) == 1,
    ]
    u_mat = np.vstack(
        (
            H_u[: dimensions.T_past * dimensions.n_act, :],
            H_u[dimensions.T_past * dimensions.n_act :, :],
            H_y[: dimensions.T_past * dimensions.n_obs, :],
            np.ones((1, H_u.shape[-1])),
        )
    )
    pi = np.linalg.pinv(u_mat) @ u_mat

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
            "g_pi": cp.sum_squares((np.eye(pi.shape[0]) - pi) @ g),
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
    T = int(np.shape(w)[0])  # number of timesteps
    d = int(np.shape(w)[1])  # dimension of the signal
    if L > T:
        raise ValueError(f"L {L} must be smaller than T {T}")

    H = np.zeros((L * d, T - L + 1))
    w_vec = w.reshape(-1)
    for i in range(0, T - L + 1):
        H[:, i] = w_vec[d * i : d * (L + i)]
    return H
