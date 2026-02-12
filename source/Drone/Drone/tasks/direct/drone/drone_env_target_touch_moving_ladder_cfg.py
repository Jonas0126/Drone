from __future__ import annotations

from isaaclab.utils import configclass

from .drone_env_target_touch_moving_fast_cfg import DroneTargetTouchMovingFastEnvCfg


@configclass
class DroneTargetTouchMovingFasterEnvCfg(DroneTargetTouchMovingFastEnvCfg):
    """速度階梯第 2 階：比 Fast 更快。"""

    episode_length_s = 20.0
    moving_target_speed = 1.4


@configclass
class DroneTargetTouchMovingVeryFastEnvCfg(DroneTargetTouchMovingFastEnvCfg):
    """速度階梯第 3 階：更高速度。"""

    episode_length_s = 20.0
    moving_target_speed = 2.8


@configclass
class DroneTargetTouchMovingUltraFastEnvCfg(DroneTargetTouchMovingFastEnvCfg):
    """速度階梯第 4 階：目前最高速度。"""

    episode_length_s = 20.0
    moving_target_speed = 5.6
