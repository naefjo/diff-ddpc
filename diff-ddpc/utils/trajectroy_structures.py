from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryData:
    """
    Collect Trajectory data in a common format.
    Expects dimensions obs: (N,n_obs) and act: (N,n_action) respectively
    """

    obs_trajectory: NDArray
    act_trajectory: NDArray

    def __post_init__(
        self,
    ):
        self.obs_trajectory = np.atleast_2d(self.obs_trajectory)
        self.act_trajectory = np.atleast_2d(self.act_trajectory)
        assert (
            np.atleast_2d(self.obs_trajectory).shape[0]
            == np.atleast_2d(self.act_trajectory).shape[0]
        ), (
            f"encountered shapes obs: {self.obs_trajectory.shape},"
            f" action: {self.act_trajectory.shape}."
            "Expected (N, n_obs), (N, n_action)"
        )


@dataclass
class TrajectoryDataSet:
    """
    Collects multiple Trajectories into a single data set.
    """

    dataset: List[TrajectoryData]
    dataset_name: str

    def __add__(self, other):
        new_dataset = deepcopy(self)
        new_dataset.dataset.extend(other.dataset)
        new_dataset.dataset_name += f"_plus_{other.dataset_name}"
        return new_dataset
