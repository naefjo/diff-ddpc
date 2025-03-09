import array_api_compat
import cvxpy as cp

from ..structures import DDPCDimensions, TrajectoryDataSet
from .hankel_predictor import generate_hankel_matrices
from .structures import Model


def generate_multistep_predictor_matrix(data: TrajectoryDataSet, dims: DDPCDimensions):
    xp = array_api_compat.get_namespace(data.dataset[0].obs_trajectory)
    tp = dims.T_past
    tf = dims.T_fut
    p = dims.n_obs
    m = dims.n_act
    H_u, H_y = generate_hankel_matrices(data, dims)

    Y_p, Y_f = xp.vsplit(H_y, [tp * p])
    U_p, U_f = xp.vsplit(H_u, [tp * m])

    Phi = xp.zeros((tf * p, (tp + tf) * (m + p)))
    for i in range(tf):
        # (tp*m + tp*p + i*m)
        Z_lk = xp.vstack((U_p, Y_p, U_f[: (i + 1) * m, :]))
        Y_rpk = Y_f[i * p : (i + 1) * p, :]
        Phi[i * p : (i + 1) * p, : tp * (m + p) + (i + 1) * m] = Y_rpk @ xp.linalg.pinv(
            Z_lk
        )

    Phi_p = Phi[:, : tp * (m + p)]
    Phi_u = Phi[:, tp * (m + p) : tp * (m + p) + tf * m]
    Phi_y = Phi[:, tp * (m + p) + tf * m :]
    eye_min_phi_y_inv = xp.linalg.inv(xp.eye(Phi.shape[0]) - Phi_y)
    return xp.hstack((eye_min_phi_y_inv @ Phi_p, eye_min_phi_y_inv @ Phi_u))


def generate_multistep_predictor(
    data: TrajectoryDataSet, dims: DDPCDimensions
) -> Model:
    multi_step_predictor = generate_multistep_predictor_matrix(data, dims)

    u_past = cp.Parameter((dims.T_past, dims.n_act))
    y_past = cp.Parameter((dims.T_past, dims.n_obs))
    u = cp.Variable((dims.T_fut, dims.n_act))
    y = cp.Variable((dims.T_fut, dims.n_obs))

    u_p_size = dims.T_past * dims.n_act
    y_p_size = dims.T_past * dims.n_obs
    Phi_u_p = multi_step_predictor[:, :u_p_size]
    Phi_y_p = multi_step_predictor[:, u_p_size : u_p_size + y_p_size]
    Phi_u_f = multi_step_predictor[:, u_p_size + y_p_size :]
    constraints = [
        y.reshape((-1, 1))
        == Phi_u_p @ u_past.reshape((-1, 1))
        + Phi_y_p @ y_past.reshape((-1, 1))
        + Phi_u_f @ u.reshape((-1, 1))
    ]

    return Model(
        constraints,
        {
            "u_past": u_past,
            "y_past": y_past,
            "u": u,
            "y": y,
        },
        dict(),
    )
