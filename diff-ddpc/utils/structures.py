from dataclasses import dataclass


@dataclass
class DDPCDimensions:
    T_past: int
    T_fut: int
    n_obs: int
    n_act: int
