from __future__ import annotations

from isaaclab.utils import configclass

from ..drone_env_target_touch_moving_cfg import DroneTargetTouchMovingEnvCfg
from .base_cfg import DroneTargetTouchVehicleBaseEnvCfg


@configclass
class DroneTargetTouchVehicleStage0EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage0：固定目標距離範圍 [10, 40] 公尺（不啟用課程）。"""

    # 與 Stage1 同步（只保留目標範圍不同）
    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -10.0
    spawn_xy_max = 10.0
    death_penalty = 200.0
    touch_bonus_reward = 200.0
    approach_reward_scale = 0.2
    distance_penalty_scale = 0.2
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    # Stage0 固定採用原先課程第 1 階段範圍，不做課程切換。
    target_spawn_distance_min = 10.0
    target_spawn_distance_max = 40.0
    # Stage0 不使用任何 target-distance 課程設定。
    target_distance_curriculum_enabled = False
    far_away_penalty = 200.0
    far_away_termination_distance = 80.0
    distance_to_goal_tanh_scale = 1.6


@configclass
class DroneTargetTouchVehicleEnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage1：參數對齊 Stage2（目標重生距離範圍 [20, 50] 公尺）。"""

    # 與 Stage0 同步（僅目標距離範圍不同）
    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    # Stage1 對齊 Stage2：加重死亡失敗成本，降低高風險策略比例。
    death_penalty = 300.0
    touch_bonus_reward = 200.0
    # Stage1 對齊 Stage2：降低接近速度獎勵，避免高風險衝刺。
    approach_reward_scale = 0.1
    distance_penalty_scale = 0.2
    time_penalty_scale = 0.20
    tcmd_lambda_4 = 5e-3
    tcmd_lambda_5 = 5e-4
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 50.0
    far_away_termination_distance = 80.0
    distance_to_goal_tanh_scale = 3.2
    failure_penalty = 300.0


@configclass
class DroneTargetTouchVehicleStage2EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage2：參數與 Stage1 一致，僅目標距離範圍為 [20, 50] 公尺。"""

    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    # Stage2 加重死亡失敗成本，降低高風險策略比例。
    death_penalty = 300.0
    touch_bonus_reward = 200.0
    # Stage2 進一步降低接近速度獎勵，避免以高風險衝刺換取短期回饋。
    approach_reward_scale = 0.1
    distance_penalty_scale = 0.2
    # Stage2 每步時間成本略增，鼓勵更快碰觸。
    time_penalty_scale = 0.20
    # Stage2 進一步加重控制命令懲罰，強化平滑穩定控制。
    tcmd_lambda_4 = 5e-3
    tcmd_lambda_5 = 5e-4
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    # Stage2 fine-tune 目標：在既有策略上擴展到更遠重生距離。
    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 50.0
    far_away_termination_distance = 80.0
    # Stage2 距離映射尺度（對齊 2026-03-03_15-33-30 訓練配置）。
    distance_to_goal_tanh_scale = 28.0
    # Stage2 failed_no_touch 懲罰加重。
    failure_penalty = 300.0


@configclass
class DroneTargetTouchVehicleStage3MovingEnvCfg(DroneTargetTouchMovingEnvCfg):
    """Vehicle touch Stage3：移動目標版本（承接 Stage1/2 已收斂策略）。"""

    # 與 Vehicle 系列對齊：使用 25 維擴展觀測。
    observation_space = 25
    use_extended_observation = True
    # 重生規則與 Stage1/2 對齊（單次 reset 分佈一致，只是目標會移動）。
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    spawn_z_min = 1.0
    spawn_z_max = 5.0
    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 50.0
    # 2026-03-04_09-32-38 訓練快照對齊。
    target_distance_curriculum_enabled = False
    target_distance_curriculum_mode = "vector_steps"
    target_distance_curriculum_stages = None
    target_distance_curriculum_stage_end_steps = None

    ang_vel_reward_scale = -0.002
    distance_penalty_only_when_not_approaching = False
    terminate_on_ground_contact = False
    death_penalty = 300.0
    failure_penalty = 300.0
    far_away_penalty = 200.0
    far_away_termination_distance = 80.0
    touch_bonus_reward = 400.0
    near_touch_outer_radius = 0.75
    approach_reward_scale = 0.1
    distance_to_goal_reward_scale = 18.0
    distance_penalty_scale = 0.2
    time_penalty_scale = 0.20
    tcmd_lambda_4 = 5e-3
    tcmd_lambda_5 = 5e-4
    distance_to_goal_tanh_scale = 3.2

    # Stage3 目標會移動：使用 5 m/s。
    moving_target_speed = 5.0
    moving_target_turn_rate_limit = 1.2
    moving_target_no_instant_reverse = True
    moving_target_vertical_dir_scale = 0.6
    moving_target_z_wave_amplitude = 0.2
    moving_target_z_wave_period_s = 8.0
    moving_target_z_min = 0.5
    moving_target_z_max = 5.0


@configclass
class DroneTargetTouchVehicleStage3EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage3：參數與 Stage1 一致，僅目標距離範圍為 [20, 40] 公尺。"""

    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 200.0
    touch_bonus_reward = 200.0
    approach_reward_scale = 0.2
    distance_penalty_scale = 0.2
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 40.0
    far_away_termination_distance = 80.0
    distance_to_goal_tanh_scale = 1.6


@configclass
class DroneTargetTouchVehicleStage4EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage4：參數與 Stage1 一致，僅目標距離範圍為 [20, 40] 公尺。"""

    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 200.0
    touch_bonus_reward = 200.0
    approach_reward_scale = 0.2
    distance_penalty_scale = 0.2
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 40.0
    far_away_termination_distance = 80.0
    distance_to_goal_tanh_scale = 1.6


@configclass
class DroneTargetTouchVehicleStage5EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage5：參數與 Stage1 一致，僅目標距離範圍為 [25, 50] 公尺。"""

    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 200.0
    touch_bonus_reward = 200.0
    approach_reward_scale = 0.2
    distance_penalty_scale = 0.2
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 25.0
    target_spawn_distance_max = 50.0
    far_away_termination_distance = 80.0
    distance_to_goal_tanh_scale = 1.6
