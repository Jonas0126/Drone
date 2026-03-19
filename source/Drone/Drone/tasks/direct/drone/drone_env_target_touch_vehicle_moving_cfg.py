from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .config.drone import DRONE_CFG
from .drone_env_target_touch_cfg import DroneTargetTouchEnvWindow


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
class DroneTargetTouchVehicleMovingStage4TaipeiDemoTestEnvCfg(DroneTargetTouchVehicleMovingStage4TestEnvCfg):
    """Stage4 台北城鎮展示環境。"""

    # 大型城鎮 USD 預設只開單一 env，避免展示時複製多份城市模型造成負擔。
    scene: InteractiveSceneCfg = DroneTargetTouchVehicleMovingTaipeiDemoSceneCfg(
        num_envs=1, env_spacing=400, replicate_physics=True, clone_in_fabric=False
    )
    # 模型仍沿用 Stage4 原本的邏輯高度分布，只在畫面上整體抬高到城鎮上空。
    spawn_z_min = 1.0
    spawn_z_max = 5.0
    target_spawn_z_min = 0.5
    target_spawn_z_max = 5.0
    moving_target_z_min = 0.5
    moving_target_z_max = 5.0
    # 展示畫面只抬高約 55m，保留「在城鎮上空飛」的感覺，同時不要離地景太遠。
    visual_altitude_offset_z = 55.0
    # 台北展示版直接對齊 Taipei_demo_001.usd 內的雙燈設定（主光 + 天空 dome）。
    use_default_distant_light = True
    default_distant_light_prim_path = "/Environment/defaultLight"
    default_distant_light_intensity = 1000.0
    default_distant_light_exposure = 1.0
    default_distant_light_angle_deg = 1.0
    default_distant_light_color = (1.0, 1.0, 1.0)
    default_distant_light_normalize = True
    default_distant_light_euler_deg = (45.0, 0.0, 90.0)
    use_default_dome_light = True
    default_dome_light_prim_path = "/Environment/DomeLight"
    default_dome_light_intensity = 500.0
    default_dome_light_exposure = 0.0
    default_dome_light_color = (1.0, 1.0, 1.0)
    default_dome_light_texture_file = (
        "/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/assets/SubUSDs/textures/Sky_horiz_6-2048.jpg"
    )
    default_dome_light_texture_format = "latlong"
    default_dome_light_euler_deg = (-270.0, 0.0, 270.0)
    # 城市資產已整體搬到原點附近，因此展示錨點直接使用 (0, 0) 即可對齊 101 周邊。
    scene_anchor_enabled = True
    scene_anchor_prim_path = None
    scene_anchor_search_root_path = None
    scene_anchor_search_prim_name = None
    scene_anchor_fallback_xy = (0.0, 0.0)
    scene_anchor_clearance_m = 50.0
    scene_anchor_target_extra_clearance_m = 5.0
    # 台北 demo 直接用指定矩形區域重生（原始 Taipei_demo 座標約 x=58059~58660, y=117575~118393）。
    # 城市已平移 (-58800.16739, -117400.47675, 0)，因此目前世界座標對應為：
    # x=-741.16739~-140.16739, y=174.52325~992.52325。
    scene_anchor_spawn_rect_enabled = True
    scene_anchor_spawn_rect_x_min = -741.16739
    scene_anchor_spawn_rect_x_max = -140.16739
    scene_anchor_spawn_rect_y_min = 174.52325
    scene_anchor_spawn_rect_y_max = 992.52325
    # 雖然畫面上會飛到較大範圍，但模型觀測仍維持局部 x/y 範圍，避免超出 Stage4 訓練分布。
    observation_local_origin_xy_enabled = True
    # 台北展示版目前只保留「重生時不要在建物內」，先關掉移動中的提早避障。
    scene_obstacle_avoidance_enabled = False
    scene_obstacle_spawn_clearance_enabled = True
    scene_obstacle_search_root_paths = (
        "/World/envs/env_{env_id}/TaipeiCity/building_Tiles_00/xform0",
        "/World/envs/env_{env_id}/TaipeiCity/taipei101_submission",
    )
    # 台北展示版額外加大軌跡球，並替無人機加上半透明亮色外殼，讓遠景鏡頭更容易辨識本體。
    test_trail_marker_radius = 0.1
    drone_outline_enabled = True
    drone_outline_radius = 0.34
    drone_outline_color = (0.2, 1.0, 1.0)
    drone_outline_emissive_color = (0.14, 0.42, 0.42)
    drone_outline_opacity = 0.22


@configclass
class DroneTargetTouchVehicleMovingStage5TestEnvCfg(DroneTargetTouchVehicleMovingStage5EnvCfg):
    # Stage5 測試環境沿用 Stage4-Test 的可視化設定。
    test_trail_enabled = True
    test_touch_reset_delay_steps = 15


@configclass
class DroneTargetTouchVehicleMovingStage5TaipeiDemoTestEnvCfg(DroneTargetTouchVehicleMovingStage5TestEnvCfg):
    """Stage5 台北城鎮展示環境。"""

    # 與 Stage4 台北 demo 相同：單一城市場景、不複製多份，避免大型 USD 展示負擔過重。
    scene: InteractiveSceneCfg = DroneTargetTouchVehicleMovingTaipeiDemoSceneCfg(
        num_envs=1, env_spacing=400, replicate_physics=True, clone_in_fabric=False
    )
    # 模型仍沿用 Stage5 原本的邏輯高度分布，只在畫面上整體抬高到城鎮上空。
    spawn_z_min = 1.0
    spawn_z_max = 5.0
    target_spawn_z_min = 0.5
    target_spawn_z_max = 5.0
    moving_target_z_min = 0.5
    moving_target_z_max = 5.0
    # Stage5 展示畫面改抬高到約 100m，讓高速追擊時更像城市上空飛行。
    visual_altitude_offset_z = 100.0
    # 台北展示版直接對齊 Taipei_demo_001.usd 內的雙燈設定（主光 + 天空 dome）。
    use_default_distant_light = True
    default_distant_light_prim_path = "/Environment/defaultLight"
    default_distant_light_intensity = 1000.0
    default_distant_light_exposure = 1.0
    default_distant_light_angle_deg = 1.0
    default_distant_light_color = (1.0, 1.0, 1.0)
    default_distant_light_normalize = True
    default_distant_light_euler_deg = (45.0, 0.0, 90.0)
    use_default_dome_light = True
    default_dome_light_prim_path = "/Environment/DomeLight"
    default_dome_light_intensity = 500.0
    default_dome_light_exposure = 0.0
    default_dome_light_color = (1.0, 1.0, 1.0)
    default_dome_light_texture_file = (
        "/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/assets/SubUSDs/textures/Sky_horiz_6-2048.jpg"
    )
    default_dome_light_texture_format = "latlong"
    default_dome_light_euler_deg = (-270.0, 0.0, 270.0)
    # 城市資產已整體搬到原點附近，因此展示錨點直接使用 (0, 0) 即可對齊 101 周邊。
    scene_anchor_enabled = True
    scene_anchor_prim_path = None
    scene_anchor_search_root_path = None
    scene_anchor_search_prim_name = None
    scene_anchor_fallback_xy = (0.0, 0.0)
    scene_anchor_clearance_m = 50.0
    scene_anchor_target_extra_clearance_m = 5.0
    # 台北 Stage5 demo 同樣使用指定矩形區域重生（原始 Taipei_demo 座標 x=58059~58660, y=117575~118393）。
    # 城市已平移 (-58800.16739, -117400.47675, 0)，因此目前世界座標對應為：
    # x=-741.16739~-140.16739, y=174.52325~992.52325。
    scene_anchor_spawn_rect_enabled = True
    scene_anchor_spawn_rect_x_min = -741.16739
    scene_anchor_spawn_rect_x_max = -140.16739
    scene_anchor_spawn_rect_y_min = 174.52325
    scene_anchor_spawn_rect_y_max = 992.52325
    # 雖然畫面上會飛到較大範圍，但模型觀測仍維持局部 x/y 範圍，避免超出 Stage5 既有分布太多。
    observation_local_origin_xy_enabled = True
    # 台北展示版目前只保留「重生時不要在建物內」，先關掉移動中的提早避障。
    scene_obstacle_avoidance_enabled = False
    scene_obstacle_spawn_clearance_enabled = True
    scene_obstacle_search_root_paths = (
        "/World/envs/env_{env_id}/TaipeiCity/building_Tiles_00/xform0",
        "/World/envs/env_{env_id}/TaipeiCity/taipei101_submission",
    )
    # 台北展示版額外加大軌跡球，並替無人機加上半透明亮色外殼，讓遠景鏡頭更容易辨識本體。
    test_trail_marker_radius = 0.065
    drone_outline_enabled = True
    drone_outline_radius = 0.34
    drone_outline_color = (0.2, 1.0, 1.0)
    drone_outline_emissive_color = (0.14, 0.42, 0.42)
    drone_outline_opacity = 0.22
