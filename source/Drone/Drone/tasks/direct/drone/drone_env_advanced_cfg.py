# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
