from __future__ import annotations
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。

from isaaclab.utils import configclass

from .drone_env_target_touch_moving_fast_cfg import DroneTargetTouchMovingFastEnvCfg


@configclass
class DroneTargetTouchMovingFasterEnvCfg(DroneTargetTouchMovingFastEnvCfg):
    """速度階梯第 2 階：比 Fast 更快。"""

    episode_length_s = 30.0
    moving_target_speed = 5.0
    moving_target_turn_rate_limit = 1.2
    moving_target_no_instant_reverse = True
    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 50.0
    far_away_termination_distance = 1_000_000.0
    terminate_on_ground_contact = True
    ground_contact_body_margin = 0.05


@configclass
class DroneTargetTouchMovingVeryFastEnvCfg(DroneTargetTouchMovingFastEnvCfg):
    """速度階梯第 3 階：更高速度。"""

    episode_length_s = 30.0
    moving_target_speed = 10.0
    moving_target_turn_rate_limit = 1.2
    moving_target_no_instant_reverse = True
    far_away_termination_distance = 1_000_000.0
    terminate_on_ground_contact = True
    ground_contact_body_margin = 0.05


@configclass
class DroneTargetTouchMovingUltraFastEnvCfg(DroneTargetTouchMovingFastEnvCfg):
    """速度階梯第 4 階：目前最高速度。"""

    episode_length_s = 30.0
    moving_target_speed = 15.0
    moving_target_turn_rate_limit = 1.2
    moving_target_no_instant_reverse = True
    far_away_termination_distance = 1_000_000.0
    terminate_on_ground_contact = True
    ground_contact_body_margin = 0.05
