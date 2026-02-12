from __future__ import annotations

from isaaclab.utils import configclass

from .drone_env_target_touch_moving_cfg import DroneTargetTouchMovingEnvCfg


@configclass
class DroneTargetTouchMovingFastEnvCfg(DroneTargetTouchMovingEnvCfg):
    """下一階段：更快移動目標版本。

    設計目的：
    - 在 Moving 任務已收斂後，提升目標移動速度作為下一階段課程。
    - 其餘獎勵與限制沿用上一階段，降低分佈跳變。
    """

    # 相較 Moving 階段 (0.35 m/s) 提升為 2 倍。
    moving_target_speed = 0.7

    # 只在 Fast 階段移除速度懲罰。
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = -0.002

    # Fast 階段加重時間懲罰，進一步鼓勵更快完成碰觸。
    time_penalty_scale = 0.35
