from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..config.drone import DRONE_CFG
from ..target_touch.cfg import DroneTargetTouchEnvWindow


@configclass
class DroneTargetTouchVehicleMovingTaipeiDemoSceneCfg(InteractiveSceneCfg):
    """台北城鎮展示場景。"""

    # 展示版直接掛入城鎮 USD，並把整張城市平移到世界原點附近。
    # 原始 Taipei_demo_001 的 101 座標約在 (58800, 117400)，若直接讓無人機在該大座標飛行，
    # PhysX / 渲染容易出現大世界座標精度問題；因此先把城市搬回原點附近，再讓無人機用公里級半徑展示。
    taipei_city: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TaipeiCity",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-58800.16739, -117400.47675, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/assets/Taipei_demo_001.usd",
        ),
    )


@configclass
class DroneTargetTouchVehicleMovingBaseEnvCfg(DirectRLEnvCfg):
    """Vehicle-Moving 系列 touch 共同基礎設定（供 Stage0~Stage5 繼承）。"""

    # env
    episode_length_s = 120.0
    decimation = 2
    action_space = 4
    observation_space = 25
    use_extended_observation = True
    state_space = 0
    debug_vis = True
    # 測試模式軌跡預設關閉；需要的 stage 再明確打開，避免影響其他測試環境。
    test_trail_enabled = False
    # 顯示哪個 env 的軌跡（通常 play/test 只看 env 0）。
    test_trail_env_id = 0
    # 軌跡最多保留多少個取樣點；超過後採 FIFO 丟掉舊點。
    test_trail_max_points = 180
    # 每隔幾個 control step 取樣一次，避免軌跡點過密。
    test_trail_sample_steps = 2
    # 無人機軌跡取機體後方的偏移距離，讓視覺上更像「尾跡」。
    test_trail_drone_back_offset = 0.35
    # 軌跡點球體半徑；展示環境可放大，讓遠距離鏡頭下更容易看見。
    test_trail_marker_radius = 0.035
    # 測試環境碰到目標後延遲幾個 step 再 reset；0 代表不延遲。
    test_touch_reset_delay_steps = 0
    # True: 在無人機外層疊一顆半透明發光球殼，從遠景看起來像 outline / halo。
    drone_outline_enabled = False
    drone_outline_radius = 0.28
    drone_outline_color = (0.15, 0.95, 1.0)
    drone_outline_emissive_color = (0.08, 0.38, 0.42)
    drone_outline_opacity = 0.18

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
    # 展示環境可用固定視覺高度偏移，把整個飛行層抬到城市上空，但模型仍看到舊高度分布。
    visual_altitude_offset_z = 0.0
    # True: 額外補一盞 DistantLight；展示場景可用來模擬 USD 內建的太陽光。
    use_default_distant_light = False
    default_distant_light_prim_path = "/World/defaultLight"
    default_distant_light_intensity = 1000.0
    default_distant_light_exposure = 0.0
    default_distant_light_angle_deg = 1.0
    default_distant_light_color = (1.0, 1.0, 1.0)
    default_distant_light_normalize = False
    # 以 XYZ Euler 角描述 DistantLight 朝向，對齊 USD 常見的 xformOp:rotateXYZ。
    default_distant_light_euler_deg = (45.0, 0.0, 90.0)
    # True: 使用環境程式額外補上的 DomeLight；False: 交由載入場景本身決定燈光。
    use_default_dome_light = True
    default_dome_light_prim_path = "/World/Light"
    default_dome_light_intensity = 2000.0
    default_dome_light_exposure = 0.0
    default_dome_light_color = (0.75, 0.75, 0.75)
    default_dome_light_texture_file = None
    default_dome_light_texture_format = "automatic"
    default_dome_light_euler_deg = (0.0, 0.0, 0.0)
    # 展示場景可指定一個關鍵地標 prim，供重生點圍繞該地標外圍取樣。
    scene_anchor_enabled = False
    # 優先使用的精確 prim path；若地圖層級穩定，這是最快的解析方式。
    scene_anchor_prim_path = None
    # 若精確 prim path 失效，改在指定根節點下依 prim 名稱遞迴搜尋。
    scene_anchor_search_root_path = None
    scene_anchor_search_prim_name = None
    # 若連搜尋都失敗，最後才退回固定錨點座標，避免又回到 env origin 附近。
    scene_anchor_fallback_xy = None
    # 無人機與目標至少要離地標外圍多少公尺，避免出生在大型建物內。
    scene_anchor_clearance_m = 0.0
    # 若目標被推離地標安全圈，額外再留一點邊界。
    scene_anchor_target_extra_clearance_m = 0.0
    # 若指定，無人機會在錨點外圍的環狀區間內重生；未指定時退回安全圈半徑。
    scene_anchor_spawn_radius_min_m = None
    scene_anchor_spawn_radius_max_m = None
    # True: 若指定固定 XY 矩形範圍，則優先在該矩形內重生，覆蓋外圍環狀取樣。
    scene_anchor_spawn_rect_enabled = False
    scene_anchor_spawn_rect_x_min = None
    scene_anchor_spawn_rect_x_max = None
    scene_anchor_spawn_rect_y_min = None
    scene_anchor_spawn_rect_y_max = None
    # True: extended observation 的 x/y 以每回合重生點為局部原點，避免展示場景把公里級絕對座標直接送進 policy。
    observation_local_origin_xy_enabled = False
    # 展示場景可啟用建物包圍盒避障：目標快撞到建物前會沿既有轉彎規則提早轉向。
    scene_obstacle_avoidance_enabled = False
    # 展示場景可單獨啟用「重生時清出建物內部」，不影響移動中的目標路徑。
    scene_obstacle_spawn_clearance_enabled = False
    # 以這些根節點的直接子節點 bbox 當作靜態障礙物來源（通常用於城市 tile / 地標群）。
    scene_obstacle_search_root_paths = ()
    # 障礙物 bbox 在 XY 平面額外膨脹多少公尺，讓目標更早開始轉彎。
    scene_obstacle_bbox_margin_m = 6.0
    # 小於此平面尺寸的 bbox 直接忽略，避免把小 props 也當成路徑障礙。
    scene_obstacle_min_size_xy_m = 8.0
    # 小於此高度的 bbox 直接忽略，避免把道路貼圖或地表小起伏誤判成建物。
    scene_obstacle_min_height_m = 8.0
    # 沿目前前進方向至少往前看幾公尺來判斷是否要提早轉彎。
    scene_obstacle_lookahead_min_m = 25.0
    # lookahead 也可依目標速度放大：lookahead = max(min_m, speed * time_s)。
    scene_obstacle_lookahead_time_s = 2.0
    # 若一步內仍進入 bbox，將目標點推出建物外時再額外留多少邊界。
    scene_obstacle_pushout_margin_m = 1.0

    # reward scales
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = 0.0
    # 提高距離 shaping 權重，鼓勵持續拉近與目標的距離。
    distance_to_goal_reward_scale = 14.0
    # 降低接近速度獎勵，避免策略過度追求瞬時衝刺。
    approach_reward_scale = 0.05
    # 使用 progress 項鼓勵每步都縮短距離。
    progress_reward_scale = 1.0
    # False: 使用一般 prev->current progress；True: 只獎勵刷新本回合最佳距離。
    progress_reward_best_so_far_only = False
    # True: 用初始距離正規化 progress，避免長距離 episode 天然累積較多分數。
    progress_reward_normalize_by_initial_distance = False
    # progress 正規化分母下界，避免近距離 episode 被過度放大。
    progress_reward_initial_distance_min = 10.0
    speed_to_goal_reward_scale = 0.0
    tcmd_lambda_4 = 5e-4
    tcmd_lambda_5 = 5e-5
    touch_bonus_reward = 200.0
    # 早碰額外加成：越早 touch，額外獎勵越高。
    touch_early_bonus_scale = 200.0
    time_penalty_scale = 0.15
    # 改用「單一距離 shaping + progress 項」，關閉額外 distance penalty。
    distance_penalty_scale = 0.0
    distance_penalty_only_when_not_approaching = False
    death_penalty = 200.0
    # 傾角超限改走獨立終局懲罰，避免和一般死亡（過低/碰地）混在同一項。
    tilt_death_penalty = 200.0
    # 超過傾角門檻後的每步軟懲罰；0 代表關閉。
    tilt_excess_penalty_scale = 0.0
    far_away_penalty = 200.0
    failure_penalty = 200.0

    # termination / contact
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
    tilt_below_reward_ratio = 0.25

    # touch-task controls
    touch_radius = 0.48
    drone_body_sphere_enabled = True
    drone_body_sphere_radius = 0.2
    drone_body_sphere_margin = 0.23
    # 若指定，debug 紅色方塊固定使用此邊長；否則沿用實際 touch threshold。
    touch_marker_edge_length = None
    enable_touch_reward = True
    terminate_on_touch = True
    near_touch_outer_radius = 0.75
    near_touch_hover_speed_threshold = 0.12
    near_touch_vel_penalty_min_scale = 0.2
    # 在 near-touch 區域但尚未碰到時，distance shaping 的保留比例；1.0 代表不降權。
    near_touch_distance_reward_ratio = 1.0
    # 靠近目標卻低速盤旋不碰時的懲罰強度。
    near_touch_hover_penalty = 0.0
    near_touch_push_reward_scale = 0.0
    follow_behind_penalty_scale = 0.0
    follow_behind_outer_radius = 12.0
    follow_behind_min_approach_speed = 0.0

    # spawn ranges (relative to env origin)
    spawn_xy_min = 0.0
    spawn_xy_max = 0.0
    spawn_z_min = 1.0
    spawn_z_max = 5.0

    # target spawn ranges
    target_spawn_distance_min = None
    target_spawn_distance_max = None
    # 目標重生高度範圍；展示環境可覆寫到建物上空，避免一生成就在街區高度。
    target_spawn_z_min = 0.5
    target_spawn_z_max = 5.0
    target_distance_curriculum_enabled = False
    target_distance_curriculum_mode = "vector_steps"
    target_distance_curriculum_stages = None
    target_distance_curriculum_stage_end_steps = None

    # Moving 目標動態設定（由 DroneTargetTouchMovingEnv 讀取）
    # 目標移動模式：
    # - "evade_drone": 以遠離無人機方向為主（既有預設）
    # - "random_straight": 直線段為主，定時換向（更接近一般車輛路徑）
    # - "road_like": 多數直行，偶爾平滑連續轉彎（像道路駕駛）
    moving_target_motion_mode = "evade_drone"
    # 目標每秒移動速度（m/s）。
    moving_target_speed = 1.0
    # 若設定 min/max，則目標速度在此區間內隨機抽樣（每次 reset，或每次換向）。
    moving_target_speed_min = None
    moving_target_speed_max = None
    # road_like 模式下，轉彎時相對於直行基準速度的倍率。
    moving_target_turn_speed_ratio = 0.75
    # 目標遠離方向中 Z 分量縮放，越小越偏水平移動。
    moving_target_vertical_dir_scale = 0.6
    # 目標單步最大轉向角速度上限（rad/s），0 代表不限制。
    moving_target_turn_rate_limit = 0.0
    # True: 禁止目標單步瞬間反向。
    moving_target_no_instant_reverse = False
    # 目標 Z 方向正弦擾動幅度（m/s）。
    moving_target_z_wave_amplitude = 0.2
    # 目標 Z 方向正弦擾動週期（秒）。
    moving_target_z_wave_period_s = 8.0
    # random_straight 模式下，每段直線維持時間（秒）範圍。
    moving_target_heading_hold_min_s = 2.0
    moving_target_heading_hold_max_s = 5.0
    # random_straight 模式下，每次換向最小轉角（度）。
    moving_target_heading_min_turn_deg = 0.0
    # random_straight 模式下，每次換向最大轉角（度）。
    moving_target_heading_max_turn_deg = 180.0
    # 測試模式可指定固定目標速度（m/s）；None 代表沿用 stage 預設抽樣/固定邏輯。
    test_fixed_target_speed = None
    # road_like 模式下，多久重新決策一次「直行或轉彎」（秒）。
    moving_target_turn_decision_min_s = 2.0
    moving_target_turn_decision_max_s = 5.0
    # road_like 模式下，若進入轉彎段，轉彎本身持續時間（秒）範圍。
    moving_target_turn_segment_min_s = 2.0
    moving_target_turn_segment_max_s = 5.0
    # road_like 模式下，保持直行的機率（其餘機率會進入平滑轉彎段）。
    moving_target_straight_prob = 0.6
    # road_like 模式下，轉彎角速度範圍（度/秒）。
    moving_target_turn_rate_min_deg_s = 8.0
    moving_target_turn_rate_max_deg_s = 20.0
    # 目標高度夾限，避免鑽地或飛太高。
    moving_target_z_min = 0.5
    moving_target_z_max = 5.0
