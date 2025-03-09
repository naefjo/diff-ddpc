import diff_ddpc
import numpy as np
import torch

dims = diff_ddpc.DDPCDimensions(2, 10, 4, 2)
obs = torch.rand((100, dims.n_obs))
act = torch.rand((100, dims.n_act))

dataset = diff_ddpc.TrajectoryDataSet(
    [diff_ddpc.TrajectoryData(obs, act)],
    "test",
)

diff_ddpc.predictors.generate_hankel_predictor(
    dataset,
    dims,
    True,
)

diff_ddpc.predictors.generate_multistep_predictor(dataset, dims)

obs_np = np.random.random((100, dims.n_obs))
act_np = np.random.random((100, dims.n_act))

diff_ddpc.predictors.generate_hankel_predictor(
    diff_ddpc.TrajectoryDataSet([diff_ddpc.TrajectoryData(obs_np, act_np)], "test_np"),
    dims,
    True,
)
