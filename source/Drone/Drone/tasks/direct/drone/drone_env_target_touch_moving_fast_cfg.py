from __future__ import annotations
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。

from isaaclab.utils import configclass

from .drone_env_target_touch_moving_cfg import DroneTargetTouchMovingEnvCfg


@configclass
class DroneTargetTouchMovingFastEnvCfg(DroneTargetTouchMovingEnvCfg):
    """下一階段：更快移動目標版本。

    設計目的：
    - 在 Moving 任務已收斂後，提升目標移動速度作為下一階段課程。
    - 其餘獎勵與限制沿用上一階段，降低分佈跳變。
    """

    # Align with current Moving stage speed.
    moving_target_speed = 2.0

    # 只在 Fast 階段移除速度懲罰。
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = -0.002

    # Align reward shaping with hover baseline; keep only target-speed increase.
    time_penalty_scale = 0.15
