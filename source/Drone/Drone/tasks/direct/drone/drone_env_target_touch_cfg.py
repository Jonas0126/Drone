from __future__ import annotations
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


##
# 預先定義資產設定
##
from .config.drone import DRONE_CFG  # isort: skip


class DroneTargetTouchEnvWindow(BaseEnvWindow):
    """Target-touch 環境的視窗與除錯面板管理器。"""

    def __init__(self, env: "DroneTargetTouchEnv", window_name: str = "IsaacLab"):
        """初始化視窗與 debug UI 元件。

        Args:
            env: 環境物件。
            window_name: 視窗名稱，預設為 ``IsaacLab``。
        """
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class DroneTargetTouchEnvCfg(DirectRLEnvCfg):
    # 環境基本參數
    episode_length_s = 30.0
    decimation = 2
    action_space = 4
    observation_space = 12
    # 啟用時使用擴展觀測向量：
    # [root_pos_w(3), root_lin_vel_w(3), root_ang_vel_w(3), rot_mat(9), desired_pos_w(3), last_action(4)] = 25
    use_extended_observation = False
    state_space = 0
    debug_vis = True

    ui_window_class_type = DroneTargetTouchEnvWindow

    # 模擬參數
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

    # 場景參數
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=8, replicate_physics=True, clone_in_fabric=True
    )

    # 機體控制參數
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 4
    moment_scale = 0.05

    # 獎勵係數
    # 可選替代獎勵：r = r_prog + r_cmd - r_crash
    use_progress_cmd_crash_reward = False
    reward_progress_scale = 1.0
    reward_cmd_omega_scale = 0.0
    reward_cmd_delta_scale = 0.0
    reward_crash_height_threshold = 0.1
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = -0.002
    distance_to_goal_reward_scale = 18.0
    # 距離 shaping 的 tanh 尺度：1 - tanh(distance / scale)
    distance_to_goal_tanh_scale = 0.8
    approach_reward_scale = 1.0
    # 控制命令懲罰（r_tcmd = λ4 * ||a_omega,t|| + λ5 * ||a_t - a_{t-1}||^2）
    # 在總獎勵中以負號加入（作為懲罰）。
    tcmd_lambda_4 = 0.0
    tcmd_lambda_5 = 0.0
    touch_bonus_reward = 70.0
    time_penalty_scale = 0.15
    distance_penalty_scale = 0.2
    # True 時：只有在「未朝目標前進」時才施加距離懲罰。
    distance_penalty_only_when_not_approaching = False
    death_penalty = 30.0
    far_away_termination_distance = 20.0
    died_height_threshold = 0.3
    terminate_on_ground_contact = False
    # 可選硬式傾角終止（單位：度）。
    # 使用 projected_gravity_b 估計機體相對重力方向的傾角。
    enable_tilt_limit_termination = False
    max_tilt_deg = 35.0
    # 軟式傾角引導（僅獎勵項）：朝目標前進時偏好 [min, max] 角度區間。
    tilt_forward_reward_scale = 0.0
    tilt_target_min_deg = 30.0
    tilt_target_max_deg = 35.0
    # 區間外平滑衰減係數（越大代表區間外懲罰越弱）。
    tilt_outside_sigma_deg = 10.0
    # 低於目標下限（< tilt_target_min_deg）時的相對獎勵上限。
    # 0.0 表示中性，1.0 表示最多可與區間內同強度。
    tilt_below_reward_ratio = 0.0
    # 接地檢查高度門檻（相對 env ground 平面）。
    # 若為 None，會優先使用機體球半徑；否則回退到 died_height_threshold。
    ground_contact_height_threshold = None
    # 可取得 body 位置時的額外接地邊界。
    ground_contact_body_margin = 0.02

    # touch 任務參數
    touch_radius = 0.48
    # 啟用時，touch 成功閾值使用（目標半徑 + 機體球半徑 + margin）。
    # 將 `drone_body_sphere_radius` 設為固定值可強制指定半徑；
    # 設為 None 會由 drone USD 外接球半徑自動估計。
    drone_body_sphere_enabled = True
    drone_body_sphere_radius = 0.2
    drone_body_sphere_margin = 0.23
    enable_touch_reward = True
    terminate_on_touch = True
    near_touch_outer_radius = 0.75
    near_touch_hover_speed_threshold = 0.12
    near_touch_vel_penalty_min_scale = 0.2

    # 重生範圍（相對 env origin）
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    spawn_z_min = 1.0
    spawn_z_max = 5.0
    # 可選目標重生距離控制（相對於 drone 重生點）。
    # 若 min/max 皆設定，目標 XY 會在該距離區間內取樣。
    target_spawn_distance_min = None
    target_spawn_distance_max = None
    # 可選目標距離課程（獨立於 drone spawn 課程）。
    target_distance_curriculum_enabled = False
    target_distance_curriculum_start = None
    target_distance_curriculum_end = None
    target_distance_curriculum_ramp_steps = None
    # 可選目標距離課程線性 min/max ramp 參數。
    target_distance_curriculum_min_start = None
    target_distance_curriculum_min_end = None
    target_distance_curriculum_max_start = None
    target_distance_curriculum_max_end = None
    # 可選分段式距離課程，例如 ((5, 10), (10, 20), ...)。
    # 若設定，優先於線性 min/max ramp。
    target_distance_curriculum_stages = None
    # 可選每段課程持續步數（單位：env common steps）。
    # 例： (50000, 100000, 100000, 100000, 150000)
    # 長度需與 `target_distance_curriculum_stages` 一致。
    target_distance_curriculum_stage_steps = None
    target_distance_curriculum_print_stages = 10
    # 距離課程進度 shaping：
    # dist_frac = raw_frac ** progress_power
    # < 1.0：前期升級快、後期升級慢（困難段停留更久）
    # = 1.0：線性升級
    # > 1.0：前期升級慢、後期升級快
    target_distance_curriculum_progress_power = 1.0

    # hover 課程（spawn 範圍分段）
    curriculum_enabled = True
    curriculum_spawn_z_max_start = 2.5
    curriculum_spawn_z_max_end = 5.0
    curriculum_ramp_steps = 1_600_000
