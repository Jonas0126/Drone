# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .drone_env_basic_cfg import DroneEnvCfg


@configclass
class DroneTrainEnvCfg(DroneEnvCfg):
    """訓練用 Drone 環境設定（4 個深度相機）。"""

    # 4 個深度相機 -> 深度特徵 20 維
    observation_space = 40
    # 是否顯示目標紅方塊
    show_goal_marker = True

    # 提升深度相機解析度（保留 update_period=2）
    _BASE = DroneEnvCfg()
    depth_cam_front = _BASE.depth_cam_front.replace(height=240, width=320)
    depth_cam_back = _BASE.depth_cam_back.replace(height=240, width=320)
    depth_cam_left = _BASE.depth_cam_left.replace(height=240, width=320)
    depth_cam_right = _BASE.depth_cam_right.replace(height=240, width=320)
    del _BASE

    # curriculum level:
    # 0 -> 0 obstacles
    # 1 -> 1 obstacle
    # 2 -> 2 obstacles
    # 5 -> 5 obstacles (current hardest setting)
    curriculum_level = 5

    # training-specific reward terms for smoother and upright flight
    reward_progress_scale = 4.0
    reward_time_penalty = 0.5
    reward_ctrl_scale = 0.05
    reward_lin_vel_scale = 0.10
    reward_ang_vel_scale = 0.08
    reward_tilt_scale = 0.40
    reward_hover_scale = 1.0

    # termination thresholds
    reset_min_height = 0.35
    reset_max_height = 8.0
    max_lin_speed = 2.5
    max_ang_speed = 2.5


@configclass
class DroneTrainLevel0EnvCfg(DroneTrainEnvCfg):
    """Advanced curriculum level 0 (no obstacle)."""

    curriculum_level = 0


@configclass
class DroneTrainLevel1EnvCfg(DroneTrainEnvCfg):
    """Advanced curriculum level 1 (1 obstacle)."""

    curriculum_level = 1


@configclass
class DroneTrainLevel2EnvCfg(DroneTrainEnvCfg):
    """Advanced curriculum level 2 (2 obstacles)."""

    curriculum_level = 2


@configclass
class DroneTrainLevel3EnvCfg(DroneTrainEnvCfg):
    """Advanced curriculum level 3 (3 obstacles)."""

    curriculum_level = 3


@configclass
class DroneTrainLevel4EnvCfg(DroneTrainEnvCfg):
    """Advanced curriculum level 4 (4 obstacles)."""

    curriculum_level = 4


@configclass
class DroneTrainLevel5EnvCfg(DroneTrainEnvCfg):
    """Advanced curriculum level 5 (5 obstacles)."""

    curriculum_level = 5
