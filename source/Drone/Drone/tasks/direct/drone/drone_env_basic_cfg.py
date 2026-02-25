# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from .config.drone import DRONE_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    """自訂場景資產設定。"""
    # 自訂場景資產（USD）
    custom_scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/CustomScene",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/assets/test_006.usd",
        ),
    )


@configclass
class DroneEnvCfg(DirectRLEnvCfg):
    """Drone 環境的整體設定（模擬、場景、感測器、獎勵等）。"""
    # 環境基本參數
    episode_length_s = 30.0
    decimation = 2
    action_space = 4
    observation_space = 30
    state_space = 0
    debug_vis = False
    debug_cam = False
    debug_collision = False
    debug_depth_env_id = 2
    debug_depth_dir = "camera_debug"

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

    # 場景配置
    scene: InteractiveSceneCfg = SceneCfg(
        num_envs=64, env_spacing=50, replicate_physics=True, clone_in_fabric=False
    )

    # 機器人
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 2.5
    moment_scale = 0.05

    # 深度相機
    depth_cam_front: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/depth_cam_front",
        update_period=2,
        height=120,
        width=160,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    depth_cam_back: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/depth_cam_back",
        update_period=2,
        height=120,
        width=160,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.03, 0.0, 0.0),
            rot=(-0.5, 0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    depth_cam_left: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/depth_cam_left",
        update_period=2,
        height=120,
        width=160,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.00, 0.03, -0.1),
            rot=(0.7071, -0.7071, 0.0, 0.0),
            convention="ros",
        ),
    )

    depth_cam_right: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/depth_cam_right",
        update_period=2,
        height=120,
        width=160,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.03, -0.12),
            rot=(0.0, 0.0, -0.7071, 0.7071),
            convention="ros",
        ),
    )

    # 獎勵超參數（DroneEnv 內使用）
    reward_progress_scale = 4.0
    reward_time_penalty = 0.5
    reward_hover_scale = 1.0
    reward_ctrl_scale = 0.05
    reward_success_bonus = 30.0
    reward_collision_penalty = 50.0

    # 課程學習（障礙物方塊）
    curriculum_enabled = False
    curriculum_start_blocks = 2
    curriculum_end_blocks = -1  # -1 means all blocks
    curriculum_ramp_steps = 200_000  # total transitions (approx)
    curriculum_hidden_z = -5.0  # hide unused blocks below ground

    # 成功判定 / 穩定性門檻
    goal_radius = 0.5
    stable_lin_vel = 0.2
    stable_ang_vel = 0.5
    hover_hold_steps = 20

    # 終止距離門檻
    max_goal_distance = 15.0
