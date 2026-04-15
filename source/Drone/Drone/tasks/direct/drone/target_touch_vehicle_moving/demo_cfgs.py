from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .base_cfg import DroneTargetTouchVehicleMovingTaipeiDemoSceneCfg
from .stage_cfgs import (
    DroneTargetTouchVehicleMovingStage4TestEnvCfg,
    DroneTargetTouchVehicleMovingStage5TestEnvCfg,
)


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
