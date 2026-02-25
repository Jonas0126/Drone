# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera.tiled_camera import TiledCamera

from pxr import Gf, Usd, UsdGeom
import omni.usd

from .markers import CUBOID_MARKER_CFG, VisualizationMarkers
from .drone_env_basic_cfg import DroneEnvCfg

"""
隨機生成環境，只使用3個block,不保證會有障礙物阻擋在中間
"""
class DroneEnv(DirectRLEnv):
    """四旋翼導航與避障的直接式強化學習環境。"""
    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化環境與內部緩存。

        Args:
            cfg: 環境設定物件。
            render_mode: 渲染模式（由 DirectRLEnv 解析）。
            **kwargs: 其他傳遞給父類的參數。
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 作用在機體上的總推力與力矩
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 目標位置
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 日誌累計
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
        }

        # 取得機體索引與重量相關資訊
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # 啟用除錯視覺化
        self.set_debug_vis(self.cfg.debug_vis)

        # 深度相機除錯輸出
        self._debug_depth_step = 0
        self._debug_depth_dir_created = False
        self._debug_block_mesh_cache: dict[int, list[tuple[str, Usd.Prim]]] = {}
        self._debug_collision_hits: set[tuple[int, str]] = set()

        # 課程學習狀態
        self._curriculum_step = 0
        self._active_block_indices: list[list[int]] = [[] for _ in range(self.num_envs)]
        self._tri_block_indices: list[tuple[int, int, int] | None] = [None for _ in range(self.num_envs)]

        # --- blocks cache for solution B ---
        # 方塊 local AABB（每個 env、每個 block）
        self._block_local_aabb: list[list[tuple[Gf.Vec3d, Gf.Vec3d]]] = []
        # 方塊世界座標位置快取
        self._block_world_pos: dict[tuple[int, int], Gf.Vec3d] = {}

        # 機器人 local AABB 快取
        self._robot_local_aabb: tuple[Gf.Vec3d, Gf.Vec3d] | None = None

        self._prev_distance = torch.zeros(self.num_envs, device=self.device)
        self._hover_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


    def _setup_scene(self):
        """建立場景、機器人、感測器與方塊快取。

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

    def _pre_physics_step(self, actions: torch.Tensor):
        """在物理步進前處理動作、課程學習與除錯設定。

        Args:
            actions: Tensor，形狀 [num_envs, action_dim]，範圍預期為 [-1, 1]。

        Returns:
            None
        """
        if self.cfg.curriculum_enabled:
            # 以 transition 計數（向量化環境）
            self._curriculum_step += self.num_envs
        if self.cfg.debug_collision:
            actions = torch.zeros_like(actions)
            actions[:, 0] = -1.0
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        """將推力與力矩作用到機體上。

        Returns:
            None
        """
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """產生觀測向量（含深度相機特徵）。

        Returns:
            dict: {"policy": Tensor}，形狀 [num_envs, obs_dim]。
        """
        # ------------------------------------------------
        # 導航核心狀態
        # ------------------------------------------------
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )

        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        gravity = self._robot.data.projected_gravity_b

        # 目標距離與方向
        goal_dist = torch.linalg.norm(desired_pos_b, dim=1, keepdim=True)
        goal_dir = desired_pos_b / (goal_dist + 1e-6)

        # ------------------------------------------------
        # 深度相機特徵
        # ------------------------------------------------
        cams = {
            "front": self.scene.sensors["depth_cam_front"],
            "back": self.scene.sensors["depth_cam_back"],
        }

        depth_feats = []
        for cam in cams.values():
            depth = self._extract_camera_depth(cam)
            feats = self._extract_depth_features(depth)
            depth_feats.append(feats)

        depth_feats = torch.cat(depth_feats, dim=1)  # [num_envs, 28]

        # ------------------------------------------------
        # 組合最終 observation
        # ------------------------------------------------
        obs = torch.cat(
            [
                lin_vel,
                ang_vel,
                gravity,
                desired_pos_b,
                goal_dist,
                goal_dir,
                self._actions,     # 前一步 action
                depth_feats,
            ],
            dim=1,
        )

        observations = {"policy": obs}

        if self.cfg.debug_cam:
            self._debug_capture_depth_cams()

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """計算獎勵與更新日誌緩存。

        Returns:
            Tensor: 形狀 [num_envs] 的獎勵值。
        """
        # ------------------------------------------------
        # 距離與進度
        # ------------------------------------------------
        pos_w = self._robot.data.root_pos_w
        d = torch.linalg.norm(self._desired_pos_w - pos_w, dim=1)

        # progress = 本步距離縮短量
        progress = self._prev_distance - d
        progress = torch.clamp(progress, -1.0, 1.0)

        r_progress = 4.0 * progress

        # ------------------------------------------------
        # 時間懲罰（避免拖延/繞路）
        # ------------------------------------------------
        r_time = -0.5 * self.step_dt

        # ------------------------------------------------
        # 懸停 shaping（只在接近目標時）
        # ------------------------------------------------
        goal_radius = 0.5

        lin_v = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        ang_v = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)

        near_goal = d < goal_radius
        stable = (lin_v < 0.2) & (ang_v < 0.5)

        r_hover = torch.zeros_like(d)
        r_hover[near_goal] = torch.exp(-2.0 * lin_v[near_goal] ** 2)

        # ------------------------------------------------
        # 控制成本（平滑動作）
        # ------------------------------------------------
        r_ctrl = -0.05 * torch.sum(self._actions ** 2, dim=1)

        # ------------------------------------------------
        # 最終獎勵
        # ------------------------------------------------
        reward = r_progress + r_time + r_hover + r_ctrl

        # 更新緩存
        self._prev_distance = d.detach()

        # 日誌紀錄
        self._episode_sums["distance_to_goal"] += r_progress
        self._episode_sums["lin_vel"] += lin_v
        self._episode_sums["ang_vel"] += ang_v


        # -----------------------------
        # 強制 TensorBoard logging（除錯）
        # -----------------------------
        if not hasattr(self, "_tb_step"):
            self._tb_step = 0

        self._tb_step += self.num_envs

        if self._tb_step % 2000 == 0:
            self.extras["log"] = {
                "Debug/mean_reward": reward.mean().item(),
                "Debug/progress": r_progress.mean().item(),
                "Debug/lin_vel": lin_v.mean().item(),
                "Debug/ang_vel": ang_v.mean().item(),
                "Debug/step": self._tb_step,
            }


        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判斷終止與超時條件。

        Returns:
            Tuple[Tensor, Tensor]: (terminated, time_out)，皆為 [num_envs] 的 bool tensor。
        """
        # ------------------------------------------------
        # 超時
        # ------------------------------------------------
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # ------------------------------------------------
        # 高度安全範圍
        # ------------------------------------------------
        z = self._robot.data.root_pos_w[:, 2]
        died = (z < 0.1) | (z > 8.0)

        # ------------------------------------------------
        # 碰撞（保持現有邏輯）
        # ------------------------------------------------
        hit = self._check_drone_block_hits()

        # ------------------------------------------------
        # 過遠終止
        # ------------------------------------------------
        d = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        too_far = d > self.cfg.max_goal_distance

        # ------------------------------------------------
        # 成功：到達目標並穩定懸停
        # ------------------------------------------------
        lin_v = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        ang_v = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)

        near_goal = d < 0.5
        stable = (lin_v < 0.2) & (ang_v < 0.5)

        success_step = near_goal & stable
        self._hover_count[success_step] += 1
        self._hover_count[~success_step] = 0

        success = self._hover_count >= 20  # hold for N steps

        # ------------------------------------------------
        # 終止條件
        # ------------------------------------------------
        terminated = hit | died | success | too_far

        return terminated, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定環境，含方塊位置與目標設定。

        Args:
            env_ids: 指定要重置的環境索引；None 表示全部重置。

        Returns:
            None
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # 結束時日誌
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

        # 標準 reset 流程
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        if self.cfg.debug_collision:
            reset_env_ids = (
                set(env_ids.tolist()) if isinstance(env_ids, torch.Tensor) else set(int(e) for e in env_ids)
            )
            self._debug_collision_hits = {
                hit for hit in self._debug_collision_hits if hit[0] not in reset_env_ids
            }
            for env_id in reset_env_ids:
                self._debug_block_mesh_cache.pop(env_id, None)

        # 重設機器人狀態
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # 加上環境原點偏移
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # 重設方塊並記錄位置（solution B）
        self._reset_blocks_position(
            env_ids,
            x_range=(0, 10.0),
            y_range=(0, 10.0),
            z_range=(0.5, 8.0),
        )

        spawn_points = torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [20.0, 0.0, 0.5],
                [20.0, 20.0, 0.5],
                [20.0, 20.0, 20.0],
                [0.0, 20.0, 0.5],
                [0.0, 20.0, 20.0],
                [0.0, 0.0, 20.0],
                [20.0, 0.0, 20.0],
            ],
            device=default_root_state.device,
            dtype=default_root_state.dtype,
        )
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        for batch_idx, env_id in enumerate(env_ids_list):
            tri = self._tri_block_indices[env_id]
            if tri is not None:
                block_pos_a = self._block_world_pos.get((env_id, tri[0]))
                block_pos_b = self._block_world_pos.get((env_id, tri[1]))
            else:
                block_pos_a = None
                block_pos_b = None

            if block_pos_a is not None and block_pos_b is not None:
                side_offset = 2.0
                # 選擇軸並在靠邊時往內推
                use_x = torch.rand(1, device=default_root_state.device).item() < 0.5
                if use_x:
                    dir_a = 1.0 if float(block_pos_a[0]) < 5.0 else -1.0
                    dir_b = 1.0 if float(block_pos_b[0]) < 5.0 else -1.0
                    spawn_offset = (dir_a * side_offset, 0.0, 0.0)
                    goal_offset = (dir_b * side_offset, 0.0, 0.0)
                else:
                    dir_a = 1.0 if float(block_pos_a[1]) < 5.0 else -1.0
                    dir_b = 1.0 if float(block_pos_b[1]) < 5.0 else -1.0
                    spawn_offset = (0.0, dir_a * side_offset, 0.0)
                    goal_offset = (0.0, dir_b * side_offset, 0.0)
                spawn_z = min(float(block_pos_a[2]), 8.0)
                goal_z = min(float(block_pos_b[2]), 8.0)
                spawn_pos = torch.tensor(
                    [block_pos_a[0] + spawn_offset[0], block_pos_a[1] + spawn_offset[1], spawn_z],
                    device=default_root_state.device,
                    dtype=default_root_state.dtype,
                )
                goal_pos = torch.tensor(
                    [block_pos_b[0] + goal_offset[0], block_pos_b[1] + goal_offset[1], goal_z],
                    device=default_root_state.device,
                    dtype=default_root_state.dtype,
                )
                default_root_state[batch_idx, 0:3] = spawn_pos
                self._desired_pos_w[env_id, :] = goal_pos
            else:
                spawn_idx = torch.randint(0, spawn_points.shape[0], (1,), device=default_root_state.device)
                goal_idx = torch.randint(0, spawn_points.shape[0], (1,), device=default_root_state.device)
                default_root_state[batch_idx, 0:3] = (
                    self._terrain.env_origins[env_id] + spawn_points[spawn_idx].squeeze(0)
                )
                self._desired_pos_w[env_id, :] = (
                    self._terrain.env_origins[env_id] + spawn_points[goal_idx].squeeze(0)
                )

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        d = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        self._prev_distance[env_ids] = d
        self._hover_count[env_ids] = 0

        # -----------------------------
        # 除錯視覺化
        # -----------------------------
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

    def _debug_vis_callback(self, event):
        """視覺化回呼：更新目標點標記。

        Args:
            event: 事件物件（由框架傳入，未使用）。

        Returns:
            None
        """
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._desired_pos_w)

        # -----------------------------
        # 深度相機除錯（未改動）
        # -----------------------------
    def _extract_camera_depth(self, camera: TiledCamera) -> torch.Tensor:
        """從相機輸出中抽取深度影像張量。

        Args:
            camera: TiledCamera 物件。

        Returns:
            Tensor: 深度影像，形狀 [num_envs, H, W] 或 [num_envs, H, W, 1]。
        """
        data = camera.data
        output = data.output if hasattr(data, "output") else data
        if isinstance(output, dict) and "distance_to_image_plane" in output:
            return output["distance_to_image_plane"]
        if hasattr(output, "distance_to_image_plane"):
            return output.distance_to_image_plane
        raise KeyError("Camera output missing 'distance_to_image_plane'")

    def _depth_to_uint8(self, depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
        """將深度圖正規化並轉為 8-bit 影像。

        Args:
            depth: 深度影像陣列，形狀 [H, W]。
            depth_min: 正規化最小深度。
            depth_max: 正規化最大深度。

        Returns:
            np.ndarray: uint8 影像，形狀 [H, W]。
        """
        valid = np.isfinite(depth) & (depth > 0)
        clipped = np.clip(depth, depth_min, depth_max)
        clipped[~valid] = depth_max
        norm = (clipped - depth_min) / (depth_max - depth_min)
        img = (norm * 255.0).astype(np.uint8)
        return img

    def _save_png(self, img: np.ndarray, path: str) -> None:
        """將影像保存為 PNG（優先使用 imageio）。

        Args:
            img: uint8 影像陣列。
            path: 輸出檔案路徑。

        Returns:
            None
        """
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError:
            from PIL import Image
            Image.fromarray(img).save(path)
            return
        imageio.imwrite(path, img)

    def _debug_capture_depth_cams(self) -> None:
        """除錯用：儲存指定 env 的深度相機影像。

        Returns:
            None
        """
        env_id = self.cfg.debug_depth_env_id
        if env_id < 0 or env_id >= self.num_envs:
            return

        # 只建立一次輸出資料夾
        if not self._debug_depth_dir_created:
            os.makedirs(self.cfg.debug_depth_dir, exist_ok=True)
            self._debug_depth_dir_created = True

        cameras = {
            "front": self.scene.sensors["depth_cam_front"],
            "back": self.scene.sensors["depth_cam_back"],
        }
        depth_min, depth_max = self.cfg.depth_cam_front.spawn.clipping_range

        for name, camera in cameras.items():
            depth = self._extract_camera_depth(camera)
            depth_env = depth[env_id].squeeze().detach().cpu().numpy()
            img = self._depth_to_uint8(depth_env, depth_min, depth_max)
            filename = f"env{env_id}_{name}_step{self._debug_depth_step:06d}.png"
            path = os.path.join(self.cfg.debug_depth_dir, filename)
            self._save_png(img, path)

        self._debug_depth_step += 1

    # -----------------------------
    # Mesh/AABB 輔助函式（solution B）
    # -----------------------------
    def _find_first_mesh(self, prim: Usd.Prim) -> Usd.Prim | None:
        """從 Prim 階層遞迴尋找第一個 Mesh。

        Args:
            prim: USD Prim。

        Returns:
            Usd.Prim | None: 找到的 Mesh Prim，若無則為 None。
        """
        if prim.IsA(UsdGeom.Mesh):
            return prim
        for child in prim.GetChildren():
            mesh = self._find_first_mesh(child)
            if mesh is not None:
                return mesh
        return None

    def _compute_local_mesh_aabb(self, mesh_prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d]:
        """計算 Mesh 在 local 空間的 AABB。

        Args:
            mesh_prim: Mesh Prim。

        Returns:
            Tuple[Gf.Vec3d, Gf.Vec3d]: (min, max) 的 local AABB。
        """
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=["default", "render", "proxy"],
            useExtentsHint=True,
        )
        box = bbox_cache.ComputeLocalBound(mesh_prim).GetBox()
        return box.GetMin(), box.GetMax()

    def _aabb_overlap(self, a_min: Gf.Vec3d, a_max: Gf.Vec3d, b_min: Gf.Vec3d, b_max: Gf.Vec3d) -> bool:
        """判斷兩個 AABB 是否重疊。

        Args:
            a_min: A 的最小角。
            a_max: A 的最大角。
            b_min: B 的最小角。
            b_max: B 的最大角。

        Returns:
            bool: 是否重疊。
        """
        return (
            a_min[0] <= b_max[0]
            and a_max[0] >= b_min[0]
            and a_min[1] <= b_max[1]
            and a_max[1] >= b_min[1]
            and a_min[2] <= b_max[2]
            and a_max[2] >= b_min[2]
        )

    def _compute_robot_world_aabb(self, env_id: int) -> tuple[Gf.Vec3d, Gf.Vec3d]:
        """計算機器人在世界座標的 AABB（忽略旋轉）。

        Args:
            env_id: 環境索引。

        Returns:
            Tuple[Gf.Vec3d, Gf.Vec3d]: (min, max) 世界座標 AABB。
        """
        # lazy init：第一次用到時才抓 USD local AABB
        if self._robot_local_aabb is None:
            stage = omni.usd.get_context().get_stage()
            robot_mesh_path = f"/World/envs/env_{env_id}/Robot/body/body"
            robot_mesh = stage.GetPrimAtPath(robot_mesh_path)
            if not robot_mesh.IsValid():
                # scene 還沒 ready，回傳一個空 AABB，避免 crash
                zero = Gf.Vec3d(0.0, 0.0, 0.0)
                return zero, zero
            self._robot_local_aabb = self._compute_local_mesh_aabb(robot_mesh)

        local_min, local_max = self._robot_local_aabb
        pos = self._robot.data.root_pos_w[env_id]

        # AABB（忽略 rotation，debug 足夠）
        return (
            Gf.Vec3d(
                float(pos[0] + local_min[0]),
                float(pos[1] + local_min[1]),
                float(pos[2] + local_min[2]),
            ),
            Gf.Vec3d(
                float(pos[0] + local_max[0]),
                float(pos[1] + local_max[1]),
                float(pos[2] + local_max[2]),
            ),
        )

    # ---------------------------
    # Block cache + lazy init
    # ---------------------------

    def _ensure_block_cache(self, env_id: int) -> None:
        """確保指定 env 的方塊快取已建立。

        Args:
            env_id: 環境索引。

        Returns:
            None
        """
        # ---- 保證外層 list 長度正確（關鍵修正）----
        if not hasattr(self, "_blocks_per_env") or self._blocks_per_env is None:
            self._blocks_per_env = []
        if not hasattr(self, "_block_local_aabb") or self._block_local_aabb is None:
            self._block_local_aabb = []

        # 擴充到 num_envs
        while len(self._blocks_per_env) < self.num_envs:
            self._blocks_per_env.append([])
        while len(self._block_local_aabb) < self.num_envs:
            self._block_local_aabb.append([])

        # 已經 cache 過，直接回傳
        if self._blocks_per_env[env_id]:
            return

        # ---- 正式從 USD 讀取 ----
        stage = omni.usd.get_context().get_stage()
        env_scene_path = f"/World/envs/env_{env_id}/CustomScene/Blocks"
        env_scene_prim = stage.GetPrimAtPath(env_scene_path)
        if not env_scene_prim.IsValid():
            # scene 尚未 ready（合法情況）
            return

        blocks: list[UsdGeom.Xform] = []
        aabbs: list[tuple[Gf.Vec3d, Gf.Vec3d]] = []

        for child in env_scene_prim.GetChildren():
            if not child.IsA(UsdGeom.Xform):
                continue

            mesh = self._find_first_mesh(child)
            if mesh is None:
                continue

            blocks.append(UsdGeom.Xform(child))
            aabbs.append(self._compute_local_mesh_aabb(mesh))

        # 保證一對一
        if len(blocks) != len(aabbs):
            raise RuntimeError(
                f"[BlockCache] env {env_id}: blocks({len(blocks)}) != aabbs({len(aabbs)})"
            )

        self._blocks_per_env[env_id] = blocks
        self._block_local_aabb[env_id] = aabbs

    def _collect_blocks(self):
        """初始化方塊快取結構（lazy）。

        Returns:
            None
        """
        # 不要在這裡直接遍歷 USD；先建立空快取，真正工作延後到 lazy init
        self._blocks_per_env = [[] for _ in range(self.num_envs)]
        self._block_local_aabb = [[] for _ in range(self.num_envs)]
        self._block_world_pos = {}
        self._active_block_indices = [[] for _ in range(self.num_envs)]
        self._tri_block_indices = [None for _ in range(self.num_envs)]

    def _get_curriculum_active_count(self, total_blocks: int) -> int:
        """依課程學習設定計算可用方塊數量。

        Args:
            total_blocks: 總方塊數量。

        Returns:
            int: 目前啟用的方塊數。
        """
        if not self.cfg.curriculum_enabled or total_blocks <= 0:
            return total_blocks
        start = max(0, min(self.cfg.curriculum_start_blocks, total_blocks))
        end = self.cfg.curriculum_end_blocks
        if end is None or end <= 0:
            end = total_blocks
        end = min(end, total_blocks)
        ramp = max(1, int(self.cfg.curriculum_ramp_steps))
        frac = min(float(self._curriculum_step) / float(ramp), 1.0)
        count = int(round(start + frac * (end - start)))
        return max(0, min(count, total_blocks))

    def _reset_blocks_position(
        self,
        env_ids,
        x_range=(0, 20.0),
        y_range=(0, 20.0),
        z_range=(0.5, 20.0),
    ):
        """重設方塊位置並同步世界座標快取。

        Args:
            env_ids: 要重設的環境索引（list 或 Tensor）。
            x_range: 位置範圍 (min, max)。
            y_range: 位置範圍 (min, max)。
            z_range: 位置範圍 (min, max)。

        Returns:
            None
        """
        # env_ids 可能是 tensor，安全迭代
        for env_id in env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids:
            self._ensure_block_cache(int(env_id))
            env_id = int(env_id)

            blocks = self._blocks_per_env[env_id]
            if not blocks:
                continue

            total_blocks = len(blocks)
            # 只保留 3 個方塊供任務使用
            tri_indices: list[int] = []
            if total_blocks >= 3:
                perm3 = torch.randperm(total_blocks, device=self.device)
                tri_indices = [int(perm3[0]), int(perm3[1]), int(perm3[2])]
                self._tri_block_indices[env_id] = (tri_indices[0], tri_indices[1], tri_indices[2])
            else:
                tri_indices = list(range(total_blocks))
                self._tri_block_indices[env_id] = None

            self._active_block_indices[env_id] = tri_indices
            active_set = set(tri_indices)

            env_origin = self._terrain.env_origins[env_id]

            tri_positions: dict[int, tuple[float, float, float]] = {}
            if len(tri_indices) == 3:
                x0 = float(torch.empty(1, device=self.device).uniform_(*x_range).item())
                y0 = float(torch.empty(1, device=self.device).uniform_(*y_range).item())
                z0 = float(torch.empty(1, device=self.device).uniform_(*z_range).item())

                x1 = float(torch.empty(1, device=self.device).uniform_(*x_range).item())
                y1 = float(torch.empty(1, device=self.device).uniform_(*y_range).item())
                z1 = float(torch.empty(1, device=self.device).uniform_(*z_range).item())

                mx = 0.5 * (x0 + x1)
                my = 0.5 * (y0 + y1)
                dx = x1 - x0
                dy = y1 - y0
                norm = math.sqrt(dx * dx + dy * dy)
                if norm < 1e-3:
                    dx = 1.0
                    dy = 0.0
                    norm = 1.0
                # 垂直偏移，避免三點共線
                px = -dy / norm
                py = dx / norm
                offset = float(torch.empty(1, device=self.device).uniform_(2.0, 4.0).item())
                bx = mx + px * offset
                by = my + py * offset
                bx = float(min(max(bx, x_range[0]), x_range[1]))
                by = float(min(max(by, y_range[0]), y_range[1]))
                bz = float(torch.empty(1, device=self.device).uniform_(*z_range).item())

                tri_positions[tri_indices[0]] = (x0, y0, z0)
                tri_positions[tri_indices[1]] = (x1, y1, z1)
                tri_positions[tri_indices[2]] = (bx, by, bz)

            for block_idx, block in enumerate(blocks):
                prim = block.GetPrim()
                prim.SetActive(True)

                xformable = UsdGeom.Xformable(block)
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()

                if block_idx in tri_positions:
                    x, y, z = tri_positions[block_idx]
                else:
                    x = float(torch.empty(1, device=self.device).uniform_(*x_range).item())
                    y = float(torch.empty(1, device=self.device).uniform_(*y_range).item())
                    if block_idx in active_set:
                        z = float(torch.empty(1, device=self.device).uniform_(*z_range).item())
                    else:
                        # 未使用的方塊藏到地面下
                        z = float(self.cfg.curriculum_hidden_z)

                translate_op.Set(Gf.Vec3f(x, y, z))

                # IMPORTANT: 同步 python 端的世界座標快取
                self._block_world_pos[(env_id, block_idx)] = Gf.Vec3d(
                    float(env_origin[0] + x),
                    float(env_origin[1] + y),
                    float(env_origin[2] + z),
                )


    def _compute_block_world_aabb(self, env_id: int, block_idx: int) -> tuple[Gf.Vec3d, Gf.Vec3d]:
        """計算指定方塊在世界座標的 AABB。

        Args:
            env_id: 環境索引。
            block_idx: 方塊索引。

        Returns:
            Tuple[Gf.Vec3d, Gf.Vec3d]: (min, max) 世界座標 AABB。
        """
        self._ensure_block_cache(env_id)

        if (
            env_id >= len(self._block_local_aabb)
            or block_idx >= len(self._block_local_aabb[env_id])
        ):
            zero = Gf.Vec3d(0.0, 0.0, 0.0)
            return zero, zero

        pos = self._block_world_pos.get((env_id, block_idx), None)
        if pos is None:
            # reset 尚未發生，不要猜 AABB
            zero = Gf.Vec3d(0.0, 0.0, 0.0)
            return zero, zero

        local_min, local_max = self._block_local_aabb[env_id][block_idx]

        return (
            Gf.Vec3d(pos[0] + local_min[0], pos[1] + local_min[1], pos[2] + local_min[2]),
            Gf.Vec3d(pos[0] + local_max[0], pos[1] + local_max[1], pos[2] + local_max[2]),
        )

    # -----------------------------
    # 以 mesh points 做碰撞偵測（除錯）
    # -----------------------------
    def _find_block_meshes(self, env_id: int) -> list[tuple[str, Usd.Prim]]:
        """尋找方塊 mesh，優先使用 aabb/mesh。

        Args:
            env_id: 環境索引。

        Returns:
            list[tuple[str, Usd.Prim]]: (block_name, mesh_prim) 列表。
        """
        cached = self._debug_block_mesh_cache.get(env_id)
        if cached is not None:
            return cached

        stage = omni.usd.get_context().get_stage()
        env_scene_path = f"/World/envs/env_{env_id}/CustomScene/Blocks"
        env_scene_prim = stage.GetPrimAtPath(env_scene_path)
        if not env_scene_prim.IsValid():
            self._debug_block_mesh_cache[env_id] = []
            return []

        meshes: list[tuple[str, Usd.Prim]] = []
        for child in env_scene_prim.GetChildren():
            block_name = child.GetName()
            preferred_mesh_path = f"{child.GetPath()}/aabb/mesh"
            preferred_mesh = stage.GetPrimAtPath(preferred_mesh_path)
            if preferred_mesh.IsValid() and preferred_mesh.IsA(UsdGeom.Mesh):
                meshes.append((block_name, preferred_mesh))
                continue

            mesh = self._find_first_mesh(child)
            if mesh is not None:
                meshes.append((block_name, mesh))

        self._debug_block_mesh_cache[env_id] = meshes
        return meshes

    def _get_mesh_points_world(self, mesh_prim: Usd.Prim, time_code: Usd.TimeCode) -> np.ndarray:
        """取得 mesh 的世界座標點集合。

        Args:
            mesh_prim: Mesh Prim。
            time_code: USD 時間碼。

        Returns:
            np.ndarray: 世界座標點集合，形狀 [N, 3]。
        """
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get(time_code)
        if not points:
            return np.zeros((0, 3), dtype=np.float64)

        xform_cache = UsdGeom.XformCache(time_code)
        transform = xform_cache.GetLocalToWorldTransform(mesh_prim)
        world_points = [transform.Transform(p) for p in points]
        return np.array(world_points, dtype=np.float64)

    def _check_drone_block_hits(self) -> torch.Tensor:
        """用 mesh AABB 判斷機體是否與方塊碰撞。

        Returns:
            Tensor: 形狀 [num_envs] 的 bool tensor。
        """
        time_code = Usd.TimeCode.Default()
        root_pos_w = self._robot.data.root_pos_w.detach().cpu().numpy()
        hit_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_id in range(self.num_envs):
            meshes = self._find_block_meshes(env_id)
            if not meshes:
                continue

            pos = root_pos_w[env_id]
            for block_name, mesh_prim in meshes:
                points = self._get_mesh_points_world(mesh_prim, time_code)
                if points.size == 0:
                    continue

                pmin = points.min(axis=0)
                pmax = points.max(axis=0)
                hit = np.all(pos >= pmin) and np.all(pos <= pmax)
                if hit:
                    hit_envs[env_id] = True
                    if (env_id, block_name) not in self._debug_collision_hits:
                        self._debug_collision_hits.add((env_id, block_name))
                        # print(f"[HIT] env_{env_id} block={block_name}")

        return hit_envs
    
    def _extract_depth_features(self, depth: torch.Tensor) -> torch.Tensor:
        """從深度圖抽取統計特徵並正規化。

        Args:
            depth: Tensor，形狀 [num_envs, H, W] 或 [num_envs, H, W, 1]。

        Returns:
            Tensor: 特徵向量，形狀 [num_envs, 5]。
        """
        """
        depth: [num_envs, H, W] or [num_envs, H, W, 1]
        return: [num_envs, 7]
        """
        if depth.ndim == 4:
            depth = depth.squeeze(-1)

        B, H, W = depth.shape

        # 攤平成一維方便做百分位
        flat = depth.view(B, -1)

        # 無效值視為最遠
        depth_min, depth_max = self.cfg.depth_cam_front.spawn.clipping_range
        valid = torch.isfinite(flat) & (flat > 0)
        flat = torch.where(valid, flat, torch.full_like(flat, depth_max))

        d_min = flat.min(dim=1).values
        d_p10 = torch.quantile(flat, 0.1, dim=1)

        # 中心與方向區塊
        h0, h1 = H // 3, 2 * H // 3
        w0, w1 = W // 3, 2 * W // 3

        center = depth[:, h0:h1, w0:w1].mean(dim=(1, 2))
        left = depth[:, :, :w0].mean(dim=(1, 2))
        right = depth[:, :, w1:].mean(dim=(1, 2))
        up = depth[:, :h0, :].mean(dim=(1, 2))
        down = depth[:, h1:, :].mean(dim=(1, 2))

        # 只保留五維特徵（去掉上下）
        feats = torch.stack([d_min, d_p10, center, left, right], dim=1)
        # 正規化到 [0, 1]
        feats = torch.clamp((feats - depth_min) / (depth_max - depth_min), 0.0, 1.0)

        return feats
