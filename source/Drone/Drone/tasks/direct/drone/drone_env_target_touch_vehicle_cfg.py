from __future__ import annotations
# 中文說明：Vehicle 系列 touch 任務獨立設定（不繼承舊系列 touch cfg）。

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .config.drone import DRONE_CFG
from .drone_env_target_touch_cfg import DroneTargetTouchEnvWindow


@configclass
class DroneTargetTouchVehicleBaseEnvCfg(DirectRLEnvCfg):
    """Vehicle 系列 touch 共同基礎設定（供 Stage0~Stage5 繼承）。"""

    # env
    episode_length_s = 30.0
    decimation = 2
    action_space = 4
    # Vehicle 系列使用擴展版世界座標觀測：
    # [root_pos_w(3), root_lin_vel_w(3), root_ang_vel_w(3), rot_mat(9), desired_pos_w(3), last_action(4)]
    observation_space = 25
    use_extended_observation = True
    state_space = 0
    debug_vis = True

    ui_window_class_type = DroneTargetTouchEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=8, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 4
    moment_scale = 0.05

    # reward scales（沿用舊版 touch reward 組成）
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = 0.0
    distance_to_goal_reward_scale = 10.0
    approach_reward_scale = 0.1
    tcmd_lambda_4 = 1e-3
    tcmd_lambda_5 = 1e-4
    touch_bonus_reward = 100.0
    time_penalty_scale = 0.15
    distance_penalty_scale = 0.1
    distance_penalty_only_when_not_approaching = True
    death_penalty = 100.0

    # termination / contact
    # Vehicle 系列啟用 far-away 失敗。
    far_away_termination_distance = 20.0
    died_height_threshold = 0.3
    terminate_on_ground_contact = True
    ground_contact_height_threshold = 0.2
    ground_contact_body_margin = 0.10

    # tilt settings
    enable_tilt_limit_termination = False
    max_tilt_deg = 35.0
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 25.0
    tilt_target_max_deg = 35.0
    tilt_outside_sigma_deg = 8.0
    # 低於目標傾角下限時的獎勵比例（相對於區間內滿分）：
    # 0.0 = 不加分；0.25 = 低於下限時只給小幅加分（最多為區間內的 25%）。
    tilt_below_reward_ratio = 0.25

    # touch-task controls
    touch_radius = 0.48
    drone_body_sphere_enabled = True
    drone_body_sphere_radius = 0.2
    drone_body_sphere_margin = 0.23
    enable_touch_reward = True
    terminate_on_touch = True
    near_touch_outer_radius = 0.75
    near_touch_hover_speed_threshold = 0.12
    near_touch_vel_penalty_min_scale = 0.2

    # spawn ranges (relative to env origin)
    # Vehicle 系列固定在 env 原點的 XY 重生（不再隨機平移）。
    spawn_xy_min = 0.0
    spawn_xy_max = 0.0
    spawn_z_min = 1.0
    spawn_z_max = 5.0

    # External curriculum: each stage is a separate environment.
    curriculum_enabled = False
    target_spawn_distance_min = None
    target_spawn_distance_max = None
    target_distance_curriculum_enabled = False
    target_distance_curriculum_start = None
    target_distance_curriculum_end = None
    target_distance_curriculum_ramp_steps = None
    target_distance_curriculum_min_start = None
    target_distance_curriculum_min_end = None
    target_distance_curriculum_max_start = None
    target_distance_curriculum_max_end = None
    target_distance_curriculum_stages = None
    target_distance_curriculum_stage_steps = None
    target_distance_curriculum_print_stages = 5
    target_distance_curriculum_progress_power = 1.0

    # keep fields for compatibility with reset logic (unused in vehicle series)
    curriculum_spawn_z_max_start = 2.5
    curriculum_spawn_z_max_end = 5.0
    curriculum_ramp_steps = 1_600_000


@configclass
class DroneTargetTouchVehicleStage0EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage0：目標重生距離範圍 [6, 10] 公尺。"""

    # 對齊 2026-02-24_14-14-02 這次 Stage0 訓練快照設定
    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 100.0
    far_away_termination_distance = 15.0
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 6.0
    target_spawn_distance_max = 10.0


@configclass
class DroneTargetTouchVehicleEnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage1：目標重生距離範圍 [15, 25] 公尺。"""

    # 與 Stage0 同步（僅目標距離範圍不同）
    observation_space = 25
    use_extended_observation = True
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    death_penalty = 100.0
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    tilt_below_reward_ratio = 0.0

    target_spawn_distance_min = 15.0
    target_spawn_distance_max = 25.0
    far_away_termination_distance = 30.0
    distance_to_goal_tanh_scale = 0.8


@configclass
class DroneTargetTouchVehicleStage2EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage2：目標重生距離範圍 [20, 40] 公尺。"""

    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 40.0
    far_away_termination_distance = 60.0


@configclass
class DroneTargetTouchVehicleStage3EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage3：目標重生距離範圍 [15, 30] 公尺。"""

    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 40.0
    far_away_termination_distance = 60.0


@configclass
class DroneTargetTouchVehicleStage4EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage4：目標重生距離範圍 [20, 40] 公尺。"""

    target_spawn_distance_min = 20.0
    target_spawn_distance_max = 40.0
    far_away_termination_distance = 60.0


@configclass
class DroneTargetTouchVehicleStage5EnvCfg(DroneTargetTouchVehicleBaseEnvCfg):
    """Vehicle touch Stage5：目標重生距離範圍 [25, 50] 公尺。"""

    target_spawn_distance_min = 25.0
    target_spawn_distance_max = 50.0
    far_away_termination_distance = 70.0
