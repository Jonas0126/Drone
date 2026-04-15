from __future__ import annotations

from isaaclab.utils import configclass

from .base_cfg import DroneTargetTouchVehicleMovingBaseEnvCfg


@configclass
class DroneTargetTouchVehicleMovingStage0EnvCfg(DroneTargetTouchVehicleMovingBaseEnvCfg):
    """Vehicle-Moving Stage0：參數與 Vehicle Stage0 一致。"""

    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -10.0
    spawn_xy_max = 10.0
    death_penalty = 200.0
    touch_bonus_reward = 200.0
    # Stage0 仍保留較高接近獎勵，但較舊值下修。
    approach_reward_scale = 0.1
    distance_penalty_scale = 0.0
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 5.0
    target_spawn_distance_max = 15.0
    target_distance_curriculum_enabled = False
    far_away_penalty = 200.0
    far_away_termination_distance = 80.0
    distance_to_goal_tanh_scale = 2.4
    # Stage0 目標速度設定。
    moving_target_speed = 1.0


@configclass
class DroneTargetTouchVehicleMovingStage1EnvCfg(DroneTargetTouchVehicleMovingBaseEnvCfg):
    """Vehicle-Moving Stage1：參數與 Vehicle Stage1 一致。"""

    # Stage1 訓練回合延長到 2 分鐘（120 秒）。
    episode_length_s = 120.0
    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 300.0
    tilt_death_penalty = 300.0
    # Stage3 稍微下修距離 shaping，避免過度依賴貼近本身而弱化最後 touch 動機。
    distance_to_goal_reward_scale = 12.0
    touch_bonus_reward = 200.0
    approach_reward_scale = 0.05
    distance_penalty_scale = 0.0
    time_penalty_scale = 0.20
    # Stage1 控制命令懲罰下修，避免過度抑制追擊機動。
    tcmd_lambda_4 = 2e-3
    tcmd_lambda_5 = 2e-4
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 10.0
    target_spawn_distance_max = 20.0
    # Stage1 放寬 far-away 終止距離到 100m。
    far_away_termination_distance = 100.0
    distance_to_goal_tanh_scale = 3.2
    failure_penalty = 300.0
    # Stage1 目標速度設定。
    moving_target_speed = 3.0


@configclass
class DroneTargetTouchVehicleMovingStage2EnvCfg(DroneTargetTouchVehicleMovingBaseEnvCfg):
    """Vehicle-Moving Stage2：參數與 Vehicle Stage2 一致。"""

    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 300.0
    tilt_death_penalty = 300.0
    touch_bonus_reward = 200.0
    approach_reward_scale = 0.05
    distance_penalty_scale = 0.0
    time_penalty_scale = 0.20
    # Stage2 reward 係數對齊 Stage1，避免 reward 形狀再產生額外差異。
    tcmd_lambda_4 = 2e-3
    tcmd_lambda_5 = 2e-4
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 15.0
    target_spawn_distance_max = 25.0
    # Stage2 的目標重生距離範圍設定為 15~25m。
    far_away_termination_distance = 80.0
    distance_to_goal_tanh_scale = 3.2
    failure_penalty = 300.0
    # Stage2 目標速度設定。
    moving_target_speed = 5.0


@configclass
class DroneTargetTouchVehicleMovingStage3EnvCfg(DroneTargetTouchVehicleMovingBaseEnvCfg):
    """Vehicle-Moving Stage3：長距離慢速追擊版，目標重生距離擴大到 20~100m。"""

    # Stage3 單獨拉長回合，給長距離樣本足夠接近與修正時間。
    episode_length_s = 180.0
    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 300.0
    tilt_death_penalty = 300.0
    # Stage3 使用略大於 1m 方塊的圓形 touch 判定：0.48 + 0.20 + 0.22 = 0.90。
    drone_body_sphere_margin = 0.22
    # Stage3 的紅色方塊固定顯示為 1m 邊長，方便直觀對齊畫面判讀。
    touch_marker_edge_length = 1.0
    # Stage3 保留少量固定 touch bonus，但主要完成獎勵仍由 early bonus 主導。
    distance_to_goal_reward_scale = 12.0
    touch_bonus_reward = 50.0
    touch_early_bonus_scale = 250.0
    progress_reward_scale = 1.0
    progress_reward_best_so_far_only = False
    progress_reward_normalize_by_initial_distance = False
    approach_reward_scale = 0.05
    distance_penalty_scale = 0.0
    time_penalty_scale = 0.20
    # Stage3 reward 係數與 Stage1/Stage2 對齊。
    tcmd_lambda_4 = 2e-3
    tcmd_lambda_5 = 2e-4
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0
    # Stage3 對極端傾角重新啟用硬終止；接近側翻時直接視為死亡。
    enable_tilt_limit_termination = True
    max_tilt_deg = 85.0
    tilt_excess_penalty_scale = 10.0

    # Stage3 目標重生距離設定為 20~60m，先用較慢目標速度做穩定過渡。
    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 60.0
    # Stage3 先取消 near-touch 特化抑制，回到 shared base 的近距離設定。
    near_touch_outer_radius = 0.75
    near_touch_distance_reward_ratio = 1.0
    near_touch_hover_penalty = 0.0
    # 終止邊界比最大重生距離再放寬一段，避免長距離回合太早因 far-away 結束。
    far_away_termination_distance = 130.0
    distance_to_goal_tanh_scale = 6.0
    # Stage3 將「回合內始終留在範圍內但沒碰到」視為次一級失敗，因此比 far-away 再輕一級。
    failure_penalty = 150.0
    # Stage3 先降速，等模型穩定後再進下一階段提速。
    moving_target_speed = 3.0


@configclass
class DroneTargetTouchVehicleMovingStage4EnvCfg(DroneTargetTouchVehicleMovingStage3EnvCfg):
    """Vehicle-Moving Stage4：沿用 Stage3 設定，但目標速度提高到 6m/s。"""

    # Stage4 僅提高 moving target 速度，其餘環境與 reward 完全沿用 Stage3。
    moving_target_speed = 6.0


@configclass
class DroneTargetTouchVehicleMovingStage5EnvCfg(DroneTargetTouchVehicleMovingStage4EnvCfg):
    """Vehicle-Moving Stage5：長距離 + 車輛式平滑道路轉彎。"""

    # Stage5 擴展重生距離，提升長距離追擊難度。
    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 80.0
    # Stage5 改成每回合固定抽樣一個基準速度，避免單一固定速度過擬合。
    moving_target_speed = 10.0
    moving_target_speed_min = 3.0
    moving_target_speed_max = 12.0
    # 轉彎時自動降速，讓軌跡更像車輛過彎。
    moving_target_turn_speed_ratio = 0.7
    # Stage5 採用車輛式平滑轉彎：多數時間直行，偶爾連續轉彎。
    moving_target_motion_mode = "road_like"
    # 更像地面車輛：主要在平面移動，不做 Z 波浪擾動。
    moving_target_vertical_dir_scale = 0.0
    moving_target_z_wave_amplitude = 0.0
    # 每 4~8 秒決策一次（直行或轉彎），進一步降低轉彎頻率。
    moving_target_turn_decision_min_s = 4.0
    moving_target_turn_decision_max_s = 8.0
    # 抽到轉彎時，轉彎段縮短（短促轉向更像路口過彎）。
    moving_target_turn_segment_min_s = 1.5
    moving_target_turn_segment_max_s = 3.0
    # 60% 直行、40% 轉彎：道路風格下提供更高的轉彎發生率。
    moving_target_straight_prob = 0.6
    # 平滑轉彎角速度（度/秒）範圍再提高，讓轉彎幅度更大。
    moving_target_turn_rate_min_deg_s = 30.0
    moving_target_turn_rate_max_deg_s = 50.0
    # road_like 由角速度直接控制轉彎，不需再用方向插值限幅。
    moving_target_turn_rate_limit = 0.0
    moving_target_no_instant_reverse = True
    # 長距離 stage 提高 far-away 失敗成本。
    far_away_penalty = 300.0
    # Stage5 中等版姿態穩定：進一步抑制螺旋式前進與姿態振盪。
    lin_vel_reward_scale = -0.002
    ang_vel_reward_scale = -0.006
    # 進一步加強控制平滑懲罰，減少追擊時的高頻修正。
    tcmd_lambda_4 = 4e-3
    tcmd_lambda_5 = 4e-4
    # 中等強度前傾引導，鼓勵更一致的追車姿態。
    tilt_forward_reward_scale = 0.03


@configclass
class DroneTargetTouchVehicleMovingStage0TestEnvCfg(DroneTargetTouchVehicleMovingStage0EnvCfg):
    # Vehicle-Moving 測試環境統一使用 60 秒回合長度。
    episode_length_s = 120.0


@configclass
class DroneTargetTouchVehicleMovingStage1TestEnvCfg(DroneTargetTouchVehicleMovingStage1EnvCfg):
    # Stage1 測試環境與訓練一致，使用 120 秒回合長度。
    episode_length_s = 120.0


@configclass
class DroneTargetTouchVehicleMovingStage2TestEnvCfg(DroneTargetTouchVehicleMovingStage2EnvCfg):
    # Vehicle-Moving 測試環境統一使用 60 秒回合長度。
    episode_length_s = 120.0


@configclass
class DroneTargetTouchVehicleMovingStage3TestEnvCfg(DroneTargetTouchVehicleMovingStage3EnvCfg):
    # Stage3 測試環境與訓練環境保持一致，避免評估分布偏移。
    # 軌跡可視化只在 Stage3 測試環境啟用，避免影響其他 stage 的測試視圖。
    test_trail_enabled = True
    # Stage3 測試碰到後保留一小段時間，讓 touched 顏色切換可以被肉眼看到。
    test_touch_reset_delay_steps = 15


@configclass
class DroneTargetTouchVehicleMovingStage4TestEnvCfg(DroneTargetTouchVehicleMovingStage4EnvCfg):
    # Stage4 測試環境沿用 Stage3-Test 的可視化設定。
    test_trail_enabled = True
    test_touch_reset_delay_steps = 15


@configclass
class DroneTargetTouchVehicleMovingStage5TestEnvCfg(DroneTargetTouchVehicleMovingStage5EnvCfg):
    # Stage5 測試環境沿用 Stage4-Test 的可視化設定。
    test_trail_enabled = True
    test_touch_reset_delay_steps = 15
