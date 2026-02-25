from __future__ import annotations
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。

from isaaclab.utils import configclass

from .drone_env_target_touch_cfg import DroneTargetTouchEnvCfg, DroneTargetTouchEnvWindow


@configclass
class DroneTargetTouchMovingEnvCfg(DroneTargetTouchEnvCfg):
    """移動目標版任務設定。

    任務目標：
    - 無人機需要在目標持續移動時，仍能追上並碰觸目標點。
    - 本階段用於承接「靜態目標已訓練完成」後的下一階段課程。

    重要參數：
    - lin_vel_reward_scale / ang_vel_reward_scale:
      速度懲罰強度，值越接近 0 代表懲罰越弱、追逐更激進。
    - moving_target_speed:
      目標移動速度（m/s）。
    - moving_target_vertical_dir_scale:
      目標遠離方向中的 Z 分量權重，越小越偏水平移動。
    - moving_target_z_min / moving_target_z_max:
      目標高度安全邊界，避免移動到地板下或超過可接受高度。
    """

    ui_window_class_type = DroneTargetTouchEnvWindow

    # Disable spawn curriculum in moving stages; keep it only in static touch stage.
    curriculum_enabled = False

    # Moving 階段保留速度懲罰，避免追逐過程過度發散。
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = -0.002
    approach_reward_scale = 1.0

    # Align reward shaping with hover baseline; moving difficulty comes from target motion.
    time_penalty_scale = 0.15

    # 目標移動設定：慢速直線移動（方向由環境動態計算）。
    moving_target_speed = 1.0  # m/s
    # Z 方向縮放：控制目標「往上/往下」分量占比，避免垂直移動過大。
    moving_target_vertical_dir_scale = 0.6
    # Keep baseline moving behavior unchanged; ladder stages can override these.
    moving_target_turn_rate_limit = 0.0
    moving_target_no_instant_reverse = False
    moving_target_z_wave_amplitude = 0.2  # m/s
    moving_target_z_wave_period_s = 8.0

    # 高度夾限：保證目標不會低於地板或高於任務上限。
    moving_target_z_min = 0.5
    moving_target_z_max = 5.0
