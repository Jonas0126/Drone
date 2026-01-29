# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera.tiled_camera import TiledCamera

from pxr import Gf, UsdGeom
import omni.usd

from .drone_env_basic import DroneEnv
from .drone_env_advanced_cfg import DroneTrainEnvCfg
from .markers import CUBOID_MARKER_CFG, VisualizationMarkers


class DroneTrainEnv(DroneEnv):
    """訓練用 Drone 環境：方塊優先阻隔 spawn 與 goal，使用 4 個深度相機。"""

    cfg: DroneTrainEnvCfg

    # 方塊數量（固定使用 3 個作為阻隔）
    _BARRIER_BLOCKS = 3
    # 生成點高度（避免上下生成）
    _SPAWN_Z = 2.0
    # 生成範圍（local XY）
    _XY_MIN = 0.8
    _XY_MAX = 15.0
    # 生成點最小間距
    _SPAWN_MIN_DIST = 9.0
    # 阻隔牆寬度（沿垂直方向的偏移）
    _BARRIER_OFFSET = 0.9
    # 阻隔位置在路徑上的比例（會加隨機微擾）
    _BARRIER_T = (0.35, 0.5, 0.65)
    # 隨機微擾範圍
    _BARRIER_T_JITTER = 0.06
    _BARRIER_OFFSET_JITTER = 0.3
    # 阻隔方塊之間的最小距離（避免常常重疊）
    _BARRIER_MIN_SEP = 1.2
    _BARRIER_MAX_TRIES = 8

    def _setup_scene(self):
        """建立場景、機器人、感測器與方塊快取（含左右相機）。

        Returns:
            None
        """
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # 地形配置
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # --- 感測器 ---
        self.scene.sensors["depth_cam_front"] = TiledCamera(self.cfg.depth_cam_front)
        self.scene.sensors["depth_cam_back"] = TiledCamera(self.cfg.depth_cam_back)
        self.scene.sensors["depth_cam_left"] = TiledCamera(self.cfg.depth_cam_left)
        self.scene.sensors["depth_cam_right"] = TiledCamera(self.cfg.depth_cam_right)

        # 複製環境
        self.scene.clone_environments(copy_from_source=False)

        # CPU 模式下的碰撞過濾
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 加入光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # clone 後再收集方塊
        self._collect_blocks()

        # --- 快取機器人 local AABB（USD local 空間） ---
        stage = omni.usd.get_context().get_stage()
        robot_mesh_path = "/World/envs/env_0/Robot/body/body"
        robot_mesh = stage.GetPrimAtPath(robot_mesh_path)
        assert robot_mesh.IsValid(), f"Missing {robot_mesh_path}"
        self._robot_local_aabb = self._compute_local_mesh_aabb(robot_mesh)

    def __init__(self, cfg: DroneTrainEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化訓練環境，依設定顯示目標紅方塊。

        Args:
            cfg: 訓練環境設定物件。
            render_mode: 渲染模式（由 DirectRLEnv 解析）。
            **kwargs: 其他傳遞給父類的參數。
        """
        cfg.debug_vis = bool(getattr(cfg, "show_goal_marker", False))
        super().__init__(cfg, render_mode, **kwargs)

    def _get_observations(self) -> dict:
        """產生觀測向量（含四方向深度相機特徵）。

        Returns:
            dict: {"policy": Tensor}，形狀 [num_envs, obs_dim]。
        """
        desired_pos_b, _ = self._compute_desired_pos_b()

        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        gravity = self._robot.data.projected_gravity_b

        goal_dist = torch.linalg.norm(desired_pos_b, dim=1, keepdim=True)
        goal_dir = desired_pos_b / (goal_dist + 1e-6)

        cams = {
            "front": self.scene.sensors["depth_cam_front"],
            "back": self.scene.sensors["depth_cam_back"],
            "left": self.scene.sensors["depth_cam_left"],
            "right": self.scene.sensors["depth_cam_right"],
        }

        depth_feats = []
        for cam in cams.values():
            depth = self._extract_camera_depth(cam)
            feats = self._extract_depth_features(depth)
            depth_feats.append(feats)

        depth_feats = torch.cat(depth_feats, dim=1)  # [num_envs, 20]

        obs = torch.cat(
            [
                lin_vel,
                ang_vel,
                gravity,
                desired_pos_b,
                goal_dist,
                goal_dir,
                self._actions,
                depth_feats,
            ],
            dim=1,
        )

        observations = {"policy": obs}

        if self.cfg.debug_cam:
            self._debug_capture_depth_cams()

        return observations

    def _compute_desired_pos_b(self):
        """計算目標在機體座標系的位置。

        Returns:
            Tuple[Tensor, Tensor]: (desired_pos_b, desired_quat_b)。
        """
        from isaaclab.utils.math import subtract_frame_transforms

        return subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定環境，生成阻隔型方塊佈局與 spawn/goal。

        Args:
            env_ids: 指定要重置的環境索引；None 表示全部重置。

        Returns:
            None
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = {
            "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
            "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
            "Metrics/final_distance_to_goal": final_distance_to_goal.item(),
        }
        self.extras["log"].update(extras)

        # standard reset flow
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # base env origin
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # generate barrier blocks and spawn/goal
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        for batch_idx, env_id in enumerate(env_ids_list):
            spawn_local, goal_local = self._pick_far_spawn_goal()
            spawn_world = self._local_to_world(spawn_local, env_id)
            goal_world = self._local_to_world(goal_local, env_id)

            self._place_barrier_blocks(env_id, spawn_local, goal_local)

            default_root_state[batch_idx, 0:3] = torch.tensor(
                spawn_world,
                device=default_root_state.device,
                dtype=default_root_state.dtype,
            )
            self._desired_pos_w[env_id, :] = torch.tensor(
                goal_world,
                device=default_root_state.device,
                dtype=default_root_state.dtype,
            )

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        d = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        self._prev_distance[env_ids] = d
        self._hover_count[env_ids] = 0

    def _pick_far_spawn_goal(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """隨機挑選距離較遠的 spawn/goal（local 座標）。

        Returns:
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]: (spawn, goal)。
        """
        # 在指定範圍內隨機取點，確保距離足夠
        for _ in range(20):
            sx = float(torch.empty(1).uniform_(self._XY_MIN, self._XY_MAX).item())
            sy = float(torch.empty(1).uniform_(self._XY_MIN, self._XY_MAX).item())
            gx = float(torch.empty(1).uniform_(self._XY_MIN, self._XY_MAX).item())
            gy = float(torch.empty(1).uniform_(self._XY_MIN, self._XY_MAX).item())
            dx, dy = gx - sx, gy - sy
            if math.sqrt(dx * dx + dy * dy) >= self._SPAWN_MIN_DIST:
                return (sx, sy, self._SPAWN_Z), (gx, gy, self._SPAWN_Z)

        # fallback：對角落附近
        sx, sy = self._XY_MIN, self._XY_MIN
        gx, gy = self._XY_MAX, self._XY_MAX
        return (sx, sy, self._SPAWN_Z), (gx, gy, self._SPAWN_Z)

    def _place_barrier_blocks(
        self,
        env_id: int,
        spawn_local: tuple[float, float, float],
        goal_local: tuple[float, float, float],
    ) -> None:
        """在 spawn 與 goal 之間放置阻隔方塊。

        Args:
            env_id: 環境索引。
            spawn_local: spawn 的 local 座標。
            goal_local: goal 的 local 座標。

        Returns:
            None
        """
        self._ensure_block_cache(env_id)

        blocks = self._blocks_per_env[env_id]
        if not blocks:
            return

        total_blocks = len(blocks)
        tri_indices = self._select_barrier_indices(total_blocks)
        self._active_block_indices[env_id] = tri_indices
        if len(tri_indices) == 3:
            self._tri_block_indices[env_id] = (tri_indices[0], tri_indices[1], tri_indices[2])
        else:
            self._tri_block_indices[env_id] = None

        sx, sy, _ = spawn_local
        gx, gy, _ = goal_local
        dx, dy = gx - sx, gy - sy
        norm = math.sqrt(dx * dx + dy * dy)
        if norm < 1e-6:
            dx, dy = 1.0, 0.0
            norm = 1.0
        px, py = -dy / norm, dx / norm

        env_origin = self._terrain.env_origins[env_id]
        placed_xy: list[tuple[float, float]] = []

        for block_idx, block in enumerate(blocks):
            prim = block.GetPrim()
            prim.SetActive(True)

            xformable = UsdGeom.Xformable(block)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()

            if block_idx in tri_indices:
                best_xy = None
                for _ in range(self._BARRIER_MAX_TRIES):
                    if len(placed_xy) == 0:
                        # 第一個 block 靠近 spawn->goal 路徑，提升阻隔強度
                        t = float(torch.empty(1).uniform_(0.3, 0.7).item())
                        offset = float(torch.empty(1).uniform_(-1.0, 1.0).item())
                        bx = sx + dx * t + px * offset
                        by = sy + dy * t + py * offset
                        bx, by = self._clamp_local_xy(bx, by)
                    else:
                        # 其他 block 分散隨機
                        bx = float(torch.empty(1).uniform_(self._XY_MIN, self._XY_MAX).item())
                        by = float(torch.empty(1).uniform_(self._XY_MIN, self._XY_MAX).item())
                    if all(
                        math.hypot(bx - ox, by - oy) >= self._BARRIER_MIN_SEP for (ox, oy) in placed_xy
                    ):
                        best_xy = (bx, by)
                        break
                    best_xy = (bx, by)
                bx, by = best_xy if best_xy is not None else (self._XY_MIN, self._XY_MIN)
                bz = self._SPAWN_Z
                placed_xy.append((bx, by))
            else:
                bx, by, bz = 0.0, 0.0, float(self.cfg.curriculum_hidden_z)

            translate_op.Set(Gf.Vec3f(bx, by, bz))

            self._block_world_pos[(env_id, block_idx)] = Gf.Vec3d(
                float(env_origin[0] + bx),
                float(env_origin[1] + by),
                float(env_origin[2] + bz),
            )

    def _select_barrier_indices(self, total_blocks: int) -> list[int]:
        """隨機選擇阻隔方塊索引，避免經常連在一起。

        Args:
            total_blocks: 場景中的方塊總數。

        Returns:
            list[int]: 選中的方塊索引。
        """
        count = min(self._BARRIER_BLOCKS, total_blocks)
        if count <= 0:
            return []

        # 嘗試多次，降低相鄰索引同時出現的機率
        best = None
        best_adj = 999
        for _ in range(10):
            perm = torch.randperm(total_blocks, device=self.device).tolist()
            picks = sorted(perm[:count])
            adj = sum(1 for i in range(len(picks) - 1) if picks[i + 1] - picks[i] == 1)
            if adj == 0:
                return picks
            if adj < best_adj:
                best_adj = adj
                best = picks
        return best if best is not None else sorted(torch.randperm(total_blocks, device=self.device).tolist()[:count])

    def _clamp_local_xy(self, x: float, y: float) -> tuple[float, float]:
        """限制 local XY 在可用範圍內。

        Args:
            x: local X。
            y: local Y。

        Returns:
            tuple[float, float]: 限制後的 (x, y)。
        """
        x = min(max(x, self._XY_MIN), self._XY_MAX)
        y = min(max(y, self._XY_MIN), self._XY_MAX)
        return x, y

    def _local_to_world(self, pos: tuple[float, float, float], env_id: int) -> tuple[float, float, float]:
        """將 local 座標轉為 world 座標。

        Args:
            pos: local 座標 (x, y, z)。
            env_id: 環境索引。

        Returns:
            tuple[float, float, float]: world 座標 (x, y, z)。
        """
        env_origin = self._terrain.env_origins[env_id]
        return (
            float(env_origin[0] + pos[0]),
            float(env_origin[1] + pos[1]),
            float(env_origin[2] + pos[2]),
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """啟用或關閉目標點視覺化。

        Args:
            debug_vis: 是否啟用視覺化。

        Returns:
            None
        """
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.25, 0.25, 0.25)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
