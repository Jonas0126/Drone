# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, subtract_frame_transforms

from .drone_env_target_touch_cfg import DroneTargetTouchEnvCfg
from .markers import CUBOID_MARKER_CFG, VisualizationMarkers


class DroneTargetTouchEnv(DirectRLEnv):
    """目標碰觸基礎環境（沿用 Hover 風格的核心動力學設定）。"""

    cfg: DroneTargetTouchEnvCfg

    def __init__(self, cfg: DroneTargetTouchEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化任務狀態與快取張量。

        這裡會建立：
        - 動作/力/力矩緩衝
        - 目標位置緩衝
        - episodic reward 統計
        - 機體物理常數（質量、重力、重量）
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 當前步動作（[-1, 1]），形狀: [num_envs, 4]。
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # 上一步動作，用於計算動作變化懲罰（tcmd delta）。
        self._prev_actions = torch.zeros_like(self._actions)
        # 外力緩衝（只用 z 推力），形狀: [num_envs, 1, 3]。
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 外力矩緩衝（x/y/z），形狀: [num_envs, 1, 3]。
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 目標世界座標（每個 env 一個 3D 目標點）。
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # 展示場景可把 extended observation 的 x/y 改成「相對本回合重生點」的局部座標。
        self._observation_local_origin_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # 每次重生後，目標與無人機初始距離（供測試 print/統計）。
        self._last_respawn_target_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 上一步到目標距離（可供進度類獎勵擴充使用）。
        self._prev_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 本回合歷史最近距離：progress reward 只獎勵「刷新最佳距離」，避免來回刷分。
        self._best_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 需長期記錄到 TensorBoard 的 reward 組件名稱。
        reward_log_keys = [
            "lin_vel",
            "ang_vel",
            "distance_to_goal",
            "speed_to_goal_reward",
            "touch_bonus",
            "touch_early_bonus",
            "approach_reward",
            "progress_reward",
            "tcmd_penalty",
            "time_penalty",
            "timeout_penalty",
            "near_touch_hover_penalty",
            "near_touch_push_reward",
            "follow_behind_penalty",
            "distance_penalty",
            "death_penalty",
            "tilt_death_penalty",
            "tilt_excess_penalty",
            "tilt_forward_reward",
            "far_away_penalty",
            "failure_penalty",
        ]
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_log_keys
        }
        # 每個 env 是否在終止步達成 touched（供 reset 統計）。
        self._term_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # hover/spawn 課程累積步數（以 reset 批次累加）。
        self._curriculum_step = 0
        # 目標距離課程專用步數計數器：避免大量環境同時 reset 時課程升級過快。
        self._target_dist_curriculum_step = 0
        # 無人機等效球半徑（touch threshold / ground contact 代理用）。
        self._drone_body_sphere_radius = self._resolve_drone_body_sphere_radius()

        # body 索引（用於把外力施加到主體剛體）。
        self._body_id = self._robot.find_bodies("body")[0]
        # 機體總質量（kg）。
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        # 重力向量模長（m/s^2）。
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        # 機體重量（N = m * g），供推力映射使用。
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)

    def _resolve_drone_body_sphere_radius(self) -> float:
        """取得機體球半徑（碰觸與接地判定用）。

        優先順序：
        1. cfg 明確指定
        2. 從 USD bounding box 自動估計
        3. 失敗時回傳 0
        """
        cfg_radius = getattr(self.cfg, "drone_body_sphere_radius", None)
        if cfg_radius is not None:
            return float(cfg_radius)

        usd_path = getattr(getattr(self.cfg.robot, "spawn", None), "usd_path", None)
        if not usd_path:
            return 0.0

        try:
            from pxr import Gf, Usd, UsdGeom

            stage = Usd.Stage.Open(str(usd_path))
            if stage is None:
                return 0.0

            prim = stage.GetDefaultPrim()
            if not prim or not prim.IsValid():
                return 0.0

            bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
            world_box = bbox_cache.ComputeWorldBound(prim).GetBox()
            size = world_box.GetMax() - world_box.GetMin()
            # 包覆 AABB 的最小外接球半徑。
            radius = 0.5 * Gf.Vec3d(size[0], size[1], size[2]).GetLength()
            return max(float(radius), 0.0)
        except Exception:
            return 0.0

    def _get_touch_threshold(self) -> float:
        """回傳碰觸成功距離閾值（公尺）。"""
        if not getattr(self.cfg, "drone_body_sphere_enabled", False):
            return float(self.cfg.touch_radius)
        return float(self.cfg.touch_radius) + self._drone_body_sphere_radius + float(
            getattr(self.cfg, "drone_body_sphere_margin", 0.0)
        )

    def _get_visual_altitude_offset(self) -> float:
        """回傳展示用固定高度偏移。"""
        return float(getattr(self.cfg, "visual_altitude_offset_z", 0.0))

    def _get_logical_root_z(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """回傳扣除展示高度偏移後的機體高度。"""
        height_offset = self._get_visual_altitude_offset()
        if env_ids is None:
            return self._robot.data.root_pos_w[:, 2] - height_offset
        return self._robot.data.root_pos_w[env_ids, 2] - height_offset

    def _get_world_ground_z(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """回傳展示高度偏移後的邏輯地面高度。"""
        height_offset = self._get_visual_altitude_offset()
        if env_ids is None:
            return self._terrain.env_origins[:, 2] + height_offset
        return self._terrain.env_origins[env_ids, 2] + height_offset

    def _get_observation_anchor_offset_xy(self) -> torch.Tensor | None:
        """回傳觀測用的平面錨點偏移，避免城市大座標直接進模型。"""
        if bool(getattr(self.cfg, "observation_local_origin_xy_enabled", False)):
            return self._observation_local_origin_xy
        if not bool(getattr(self.cfg, "scene_anchor_enabled", False)):
            return None
        if self._scene_anchor_centers_w is None:
            return None
        return self._scene_anchor_centers_w[:, :2]

    def _get_ground_contact_mask(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """估計是否接地。

        判定策略：
        - 先用 root 高度與地面高度比較
        - 若可取得 body 位置，改用「任一 body 低於門檻」提升碰撞偵測穩定性
        """
        if not getattr(self.cfg, "terminate_on_ground_contact", True):
            if env_ids is None:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        if env_ids is None:
            root_z = self._robot.data.root_pos_w[:, 2]
            ground_z = self._get_world_ground_z()
        else:
            root_z = self._robot.data.root_pos_w[env_ids, 2]
            ground_z = self._get_world_ground_z(env_ids)

        cfg_thresh = getattr(self.cfg, "ground_contact_height_threshold", None)
        if cfg_thresh is None:
            if self._drone_body_sphere_radius > 0.0:
                contact_height = float(self._drone_body_sphere_radius)
            else:
                contact_height = float(getattr(self.cfg, "died_height_threshold", 0.3))
        else:
            contact_height = float(cfg_thresh)
        root_contact = root_z <= (ground_z + contact_height)

        # 若可取得各剛體位置，改用「任一 body 接近地面」作為更穩定的接地代理判定。
        body_pos_w = getattr(self._robot.data, "body_pos_w", None)
        if body_pos_w is None:
            body_state_w = getattr(self._robot.data, "body_state_w", None)
            if body_state_w is not None and body_state_w.shape[-1] >= 3:
                body_pos_w = body_state_w[..., :3]

        if body_pos_w is None:
            return root_contact

        if env_ids is None:
            min_body_z = torch.amin(body_pos_w[..., 2], dim=1)
        else:
            min_body_z = torch.amin(body_pos_w[env_ids, :, 2], dim=1)

        body_margin = float(getattr(self.cfg, "ground_contact_body_margin", 0.02))
        body_contact = min_body_z <= (ground_z + body_margin)
        return root_contact | body_contact

    def _get_tilt_exceeded_mask(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """估計是否超過傾角上限（以重力在機體座標的投影計算）。"""
        if not bool(getattr(self.cfg, "enable_tilt_limit_termination", False)):
            if env_ids is None:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        max_tilt_deg = float(getattr(self.cfg, "max_tilt_deg", 35.0))
        if env_ids is None:
            gravity_b = self._robot.data.projected_gravity_b
        else:
            gravity_b = self._robot.data.projected_gravity_b[env_ids]

        g_norm = torch.linalg.norm(gravity_b, dim=1).clamp_min(1e-6)
        cos_tilt = torch.clamp((-gravity_b[:, 2]) / g_norm, -1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
        return tilt_deg > max_tilt_deg

    def _setup_scene(self):
        """建立場景（機體、地形、環境複製與燈光）。"""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 展示場景可選擇完全沿用地圖本身的燈光設定，或在地圖燈光缺失時補上對應的 USD Lux 燈光。
        if bool(getattr(self.cfg, "use_default_distant_light", False)):
            distant_light_cfg = sim_utils.DistantLightCfg(
                intensity=float(getattr(self.cfg, "default_distant_light_intensity", 1000.0)),
                exposure=float(getattr(self.cfg, "default_distant_light_exposure", 0.0)),
                angle=float(getattr(self.cfg, "default_distant_light_angle_deg", 1.0)),
                color=tuple(getattr(self.cfg, "default_distant_light_color", (1.0, 1.0, 1.0))),
                normalize=bool(getattr(self.cfg, "default_distant_light_normalize", False)),
            )
            light_rot_deg = getattr(self.cfg, "default_distant_light_euler_deg", (45.0, 0.0, 90.0))
            light_rot_rad = torch.deg2rad(torch.tensor(light_rot_deg, dtype=torch.float))
            light_quat = quat_from_euler_xyz(
                light_rot_rad[0].unsqueeze(0), light_rot_rad[1].unsqueeze(0), light_rot_rad[2].unsqueeze(0)
            )[0]
            distant_light_cfg.func(
                getattr(self.cfg, "default_distant_light_prim_path", "/World/defaultLight"),
                distant_light_cfg,
                orientation=tuple(float(v) for v in light_quat.tolist()),
            )

        if bool(getattr(self.cfg, "use_default_dome_light", True)):
            light_cfg = sim_utils.DomeLightCfg(
                intensity=float(getattr(self.cfg, "default_dome_light_intensity", 2000.0)),
                exposure=float(getattr(self.cfg, "default_dome_light_exposure", 0.0)),
                color=tuple(getattr(self.cfg, "default_dome_light_color", (0.75, 0.75, 0.75))),
                texture_file=getattr(self.cfg, "default_dome_light_texture_file", None),
                texture_format=str(getattr(self.cfg, "default_dome_light_texture_format", "automatic")),
            )
            dome_rot_deg = getattr(self.cfg, "default_dome_light_euler_deg", (0.0, 0.0, 0.0))
            dome_rot_rad = torch.deg2rad(torch.tensor(dome_rot_deg, dtype=torch.float))
            dome_quat = quat_from_euler_xyz(
                dome_rot_rad[0].unsqueeze(0), dome_rot_rad[1].unsqueeze(0), dome_rot_rad[2].unsqueeze(0)
            )[0]
            light_cfg.func(
                getattr(self.cfg, "default_dome_light_prim_path", "/World/Light"),
                light_cfg,
                orientation=tuple(float(v) for v in dome_quat.tolist()),
            )

        self._cache_scene_anchor_data()
        self._cache_scene_obstacle_data()

    def _cache_scene_anchor_data(self):
        """快取展示地標的中心點與安全半徑。"""
        self._scene_anchor_centers_w = None
        self._scene_anchor_safe_radius = None

        if not bool(getattr(self.cfg, "scene_anchor_enabled", False)):
            return

        prim_path_template = getattr(self.cfg, "scene_anchor_prim_path", None)
        search_root_template = getattr(self.cfg, "scene_anchor_search_root_path", None)
        search_name = getattr(self.cfg, "scene_anchor_search_prim_name", None)
        fallback_xy_cfg = getattr(self.cfg, "scene_anchor_fallback_xy", None)
        if prim_path_template is None and not (search_root_template and search_name) and fallback_xy_cfg is None:
            return

        try:
            import omni.usd
            from pxr import Usd, UsdGeom
        except Exception as exc:
            print(f"[WARN][Touch] scene anchor unavailable: {exc}", flush=True)
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        centers = torch.full((self.scene.cfg.num_envs, 3), float("nan"), dtype=torch.float, device=self.device)
        safe_radii = torch.full((self.scene.cfg.num_envs,), -1.0, dtype=torch.float, device=self.device)
        clearance_m = max(float(getattr(self.cfg, "scene_anchor_clearance_m", 0.0)), 0.0)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        for env_id in range(self.scene.cfg.num_envs):
            prim_path = None
            prim = None
            if prim_path_template:
                prim_path = str(prim_path_template).format(env_id=env_id)
                prim = stage.GetPrimAtPath(prim_path)
            if (not prim or not prim.IsValid()) and search_root_template and search_name:
                search_root_path = str(search_root_template).format(env_id=env_id)
                search_root = stage.GetPrimAtPath(search_root_path)
                prim = self._find_descendant_prim_by_name(search_root, str(search_name))

            if prim and prim.IsValid():
                world_box = bbox_cache.ComputeWorldBound(prim).GetBox()
                box_min = world_box.GetMin()
                box_max = world_box.GetMax()
                center_x = 0.5 * (float(box_min[0]) + float(box_max[0]))
                center_y = 0.5 * (float(box_min[1]) + float(box_max[1]))
                center_z = 0.5 * (float(box_min[2]) + float(box_max[2]))
                dx = max(float(box_max[0]) - float(box_min[0]), 0.0)
                dy = max(float(box_max[1]) - float(box_min[1]), 0.0)
                half_diag_xy = 0.5 * ((dx * dx + dy * dy) ** 0.5)
                centers[env_id] = torch.tensor((center_x, center_y, center_z), dtype=torch.float, device=self.device)
                safe_radii[env_id] = float(half_diag_xy + clearance_m)
                continue

            if fallback_xy_cfg is not None and len(fallback_xy_cfg) >= 2:
                fallback_x = float(fallback_xy_cfg[0])
                fallback_y = float(fallback_xy_cfg[1])
                centers[env_id] = torch.tensor((fallback_x, fallback_y, 0.0), dtype=torch.float, device=self.device)
                safe_radii[env_id] = float(clearance_m)
                print(
                    f"[WARN][Touch] scene anchor fallback to fixed XY for env_{env_id}: "
                    f"({fallback_x:.5f}, {fallback_y:.5f})",
                    flush=True,
                )
                continue

            print(f"[WARN][Touch] scene anchor unresolved for env_{env_id}: {prim_path}", flush=True)

        if torch.any(safe_radii > 0.0):
            self._scene_anchor_centers_w = centers
            self._scene_anchor_safe_radius = safe_radii

    def _find_descendant_prim_by_name(self, root_prim, target_name: str):
        """在指定根節點下遞迴搜尋 prim 名稱。"""
        if root_prim is None or not root_prim.IsValid():
            return None

        target_name = str(target_name).lower()
        stack = list(root_prim.GetChildren())
        while stack:
            prim = stack.pop()
            if prim.GetName().lower() == target_name:
                return prim
            stack.extend(list(prim.GetChildren()))
        return None

    def _cache_scene_obstacle_data(self):
        """快取展示場景的靜態障礙物包圍盒（主要供 moving target 避障）。"""
        self._scene_obstacle_boxes_xy_min = [None] * self.scene.cfg.num_envs
        self._scene_obstacle_boxes_xy_max = [None] * self.scene.cfg.num_envs
        self._scene_obstacle_boxes_xy_center = [None] * self.scene.cfg.num_envs

        if not bool(getattr(self.cfg, "scene_obstacle_avoidance_enabled", False)) and not bool(
            getattr(self.cfg, "scene_obstacle_spawn_clearance_enabled", False)
        ):
            return

        root_path_templates = tuple(getattr(self.cfg, "scene_obstacle_search_root_paths", ()))
        if len(root_path_templates) == 0:
            return

        try:
            import omni.usd
            from pxr import Usd, UsdGeom
        except Exception as exc:
            print(f"[WARN][Touch] scene obstacle cache unavailable: {exc}", flush=True)
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox_margin = max(float(getattr(self.cfg, "scene_obstacle_bbox_margin_m", 0.0)), 0.0)
        min_size_xy = max(float(getattr(self.cfg, "scene_obstacle_min_size_xy_m", 0.0)), 0.0)
        min_height = max(float(getattr(self.cfg, "scene_obstacle_min_height_m", 0.0)), 0.0)

        for env_id in range(self.scene.cfg.num_envs):
            boxes_xy_min: list[tuple[float, float]] = []
            boxes_xy_max: list[tuple[float, float]] = []
            boxes_xy_center: list[tuple[float, float]] = []

            for root_template in root_path_templates:
                root_path = str(root_template).format(env_id=env_id)
                root_prim = stage.GetPrimAtPath(root_path)
                if not root_prim or not root_prim.IsValid():
                    continue

                source_prims = list(root_prim.GetChildren())
                if len(source_prims) == 0:
                    source_prims = [root_prim]

                for source_prim in source_prims:
                    if source_prim is None or not source_prim.IsValid():
                        continue
                    try:
                        world_box = bbox_cache.ComputeWorldBound(source_prim).GetBox()
                    except Exception:
                        continue
                    box_min = world_box.GetMin()
                    box_max = world_box.GetMax()
                    dx = max(float(box_max[0]) - float(box_min[0]), 0.0)
                    dy = max(float(box_max[1]) - float(box_min[1]), 0.0)
                    dz = max(float(box_max[2]) - float(box_min[2]), 0.0)
                    if max(dx, dy) < min_size_xy or dz < min_height:
                        continue

                    min_x = float(box_min[0]) - bbox_margin
                    min_y = float(box_min[1]) - bbox_margin
                    max_x = float(box_max[0]) + bbox_margin
                    max_y = float(box_max[1]) + bbox_margin
                    if not torch.isfinite(torch.tensor((min_x, min_y, max_x, max_y), dtype=torch.float)).all():
                        continue

                    boxes_xy_min.append((min_x, min_y))
                    boxes_xy_max.append((max_x, max_y))
                    boxes_xy_center.append((0.5 * (min_x + max_x), 0.5 * (min_y + max_y)))

            if len(boxes_xy_min) == 0:
                print(f"[WARN][Touch] no scene obstacle boxes cached for env_{env_id}", flush=True)
                continue

            self._scene_obstacle_boxes_xy_min[env_id] = torch.tensor(
                boxes_xy_min, dtype=torch.float, device=self.device
            )
            self._scene_obstacle_boxes_xy_max[env_id] = torch.tensor(
                boxes_xy_max, dtype=torch.float, device=self.device
            )
            self._scene_obstacle_boxes_xy_center[env_id] = torch.tensor(
                boxes_xy_center, dtype=torch.float, device=self.device
            )
            print(
                f"[INFO][Touch] cached {len(boxes_xy_min)} scene obstacle boxes for env_{env_id}",
                flush=True,
            )

    def _sample_scene_anchor_ring_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
        """在展示地標外圍的安全圈或指定環狀區間內取樣 XY。"""
        if self._scene_anchor_centers_w is None or self._scene_anchor_safe_radius is None:
            raise RuntimeError("scene anchor cache is not initialized")
        if not torch.all(self._scene_anchor_safe_radius[env_ids] > 0.0):
            raise RuntimeError("scene anchor cache is unresolved for some envs")

        theta = torch.empty((len(env_ids),), device=self.device).uniform_(0.0, 2.0 * torch.pi)
        radii = self._scene_anchor_safe_radius[env_ids]
        spawn_radius_min_cfg = getattr(self.cfg, "scene_anchor_spawn_radius_min_m", None)
        spawn_radius_max_cfg = getattr(self.cfg, "scene_anchor_spawn_radius_max_m", None)
        if spawn_radius_min_cfg is not None or spawn_radius_max_cfg is not None:
            spawn_radius_min = float(spawn_radius_min_cfg if spawn_radius_min_cfg is not None else 0.0)
            spawn_radius_max = float(spawn_radius_max_cfg if spawn_radius_max_cfg is not None else spawn_radius_min)
            if spawn_radius_min > spawn_radius_max:
                spawn_radius_min, spawn_radius_max = spawn_radius_max, spawn_radius_min
            spawn_radius_min = max(spawn_radius_min, 0.0)
            # 內半徑不能小於各 env 的安全圈半徑，避免又把出生點拉回建物內。
            min_radii = torch.clamp(self._scene_anchor_safe_radius[env_ids], min=spawn_radius_min)
            max_radii = torch.full_like(min_radii, max(spawn_radius_max, 0.0))
            max_radii = torch.maximum(max_radii, min_radii)
            if torch.any(max_radii > min_radii):
                # 用面積均勻取樣環狀區間，避免點位過度集中在內圈。
                u = torch.empty((len(env_ids),), device=self.device).uniform_(0.0, 1.0)
                radii = torch.sqrt(u * (max_radii**2 - min_radii**2) + min_radii**2)
            else:
                radii = min_radii
        x = self._scene_anchor_centers_w[env_ids, 0] + radii * torch.cos(theta)
        y = self._scene_anchor_centers_w[env_ids, 1] + radii * torch.sin(theta)
        return torch.stack((x, y), dim=1)

    def _sample_scene_anchor_spawn_xy(self, env_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """依展示場景設定取樣重生 XY；若指定矩形範圍則優先使用。"""
        if bool(getattr(self.cfg, "scene_anchor_spawn_rect_enabled", False)):
            x_min_cfg = getattr(self.cfg, "scene_anchor_spawn_rect_x_min", None)
            x_max_cfg = getattr(self.cfg, "scene_anchor_spawn_rect_x_max", None)
            y_min_cfg = getattr(self.cfg, "scene_anchor_spawn_rect_y_min", None)
            y_max_cfg = getattr(self.cfg, "scene_anchor_spawn_rect_y_max", None)
            if None not in (x_min_cfg, x_max_cfg, y_min_cfg, y_max_cfg):
                x_min = float(x_min_cfg)
                x_max = float(x_max_cfg)
                y_min = float(y_min_cfg)
                y_max = float(y_max_cfg)
                if x_min > x_max:
                    x_min, x_max = x_max, x_min
                if y_min > y_max:
                    y_min, y_max = y_max, y_min
                spawn_xy = torch.empty((len(env_ids), 2), device=self.device, dtype=dtype)
                spawn_xy[:, 0] = torch.empty((len(env_ids),), device=self.device, dtype=dtype).uniform_(x_min, x_max)
                spawn_xy[:, 1] = torch.empty((len(env_ids),), device=self.device, dtype=dtype).uniform_(y_min, y_max)
                return spawn_xy
        return self._sample_scene_anchor_ring_xy(env_ids).to(dtype=dtype)

    def _enforce_scene_anchor_clearance(
        self, points_w: torch.Tensor, env_ids: torch.Tensor, extra_clearance: float = 0.0
    ) -> torch.Tensor:
        """將點位推到展示地標安全圈外，避免生成在大型建物內。"""
        if self._scene_anchor_centers_w is None or self._scene_anchor_safe_radius is None:
            return points_w

        centers_xy = self._scene_anchor_centers_w[env_ids, :2]
        safe_radius = self._scene_anchor_safe_radius[env_ids] + max(float(extra_clearance), 0.0)
        valid_mask = safe_radius > 0.0
        if not torch.any(valid_mask):
            return points_w
        adjusted_points = points_w.clone()
        delta_xy = adjusted_points[:, :2] - centers_xy
        dist_xy = torch.linalg.norm(delta_xy, dim=1)
        inside_mask = valid_mask & (dist_xy < safe_radius)
        if not torch.any(inside_mask):
            return adjusted_points

        safe_delta = delta_xy[inside_mask]
        safe_dist = dist_xy[inside_mask]
        degenerate = safe_dist < 1e-6
        if torch.any(degenerate):
            fallback_angles = torch.empty((int(degenerate.sum().item()),), device=self.device).uniform_(0.0, 2.0 * torch.pi)
            safe_delta[degenerate, 0] = torch.cos(fallback_angles)
            safe_delta[degenerate, 1] = torch.sin(fallback_angles)
            safe_dist = torch.linalg.norm(safe_delta, dim=1)
        safe_dir = safe_delta / torch.clamp(safe_dist.unsqueeze(-1), min=1e-6)
        adjusted_points[inside_mask, 0] = centers_xy[inside_mask, 0] + safe_dir[:, 0] * safe_radius[inside_mask]
        adjusted_points[inside_mask, 1] = centers_xy[inside_mask, 1] + safe_dir[:, 1] * safe_radius[inside_mask]
        return adjusted_points

    def _pre_physics_step(self, actions: torch.Tensor):
        """把策略輸出轉成實際控制量（總推力 + 三軸力矩）。"""
        # 某些啟動/暖身流程可能短暫傳入 None，回退為零動作避免訓練中斷。
        if actions is None:
            self._actions.zero_()
        else:
            self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        """將上一階段計算好的外力/力矩套用到機體。"""
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """組裝 policy observation（預設 12 維；可切換為 extended 25 維）。"""
        if bool(getattr(self.cfg, "use_extended_observation", False)):
            rot_mat_wb = matrix_from_quat(self._robot.data.root_quat_w).reshape(self.num_envs, 9)
            root_pos_rel = self._robot.data.root_pos_w - self._terrain.env_origins
            desired_pos_rel = self._desired_pos_w - self._terrain.env_origins
            anchor_offset_xy = self._get_observation_anchor_offset_xy()
            if anchor_offset_xy is not None:
                root_pos_rel[:, :2] -= anchor_offset_xy
                desired_pos_rel[:, :2] -= anchor_offset_xy
            height_offset = self._get_visual_altitude_offset()
            if abs(height_offset) > 0.0:
                root_pos_rel[:, 2] -= height_offset
                desired_pos_rel[:, 2] -= height_offset
            obs = torch.cat(
                [
                    root_pos_rel,
                    self._robot.data.root_lin_vel_w,
                    self._robot.data.root_ang_vel_w,
                    rot_mat_wb,
                    desired_pos_rel,
                    self._actions,
                ],
                dim=-1,
            )
            return {"policy": obs}

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """計算目標碰觸任務獎勵（Moving 版本也沿用本函式）。

        任務目標：
        1. 儘快接近並碰觸目標點。
        2. 避免在目標附近「停住但不碰」。
        3. 保持可控，不要過度振盪或亂轉。

        獎勵項目說明：
        - lin_vel: 線速度平方懲罰（抑制平移過猛）。
        - ang_vel: 角速度平方懲罰（抑制姿態快速旋轉）。
        - distance_to_goal: 與目標距離 shaping，越近越高。
        - speed_to_goal_reward: 越快朝目標前進且越接近目標，獎勵越高。
        - touch_bonus: 進入 touch 半徑時的一次性正獎勵。
        - touch_early_bonus: 越早碰到目標，額外加分越多。
        - approach_reward: 在機體座標系中，朝目標方向前進速度的正獎勵。
        - time_penalty: 每一步固定扣分，鼓勵更短時間完成。
        - near_touch_hover_penalty: 靠近目標卻低速懸停、未實際碰觸時的懲罰。
        - follow_behind_penalty: 跟在目標後方但接近速度不足時的懲罰。
        """
        # 先計算會被多個 reward 項共用的狀態量，減少重複計算。
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_tanh_scale = max(float(getattr(self.cfg, "distance_to_goal_tanh_scale", 0.8)), 1e-6)
        distance_to_goal_mapped = 1.0 - torch.tanh(distance_to_goal / distance_to_goal_tanh_scale)
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        goal_dir_b = desired_pos_b / (torch.linalg.norm(desired_pos_b, dim=1, keepdim=True) + 1e-6)
        approach_speed = torch.sum(self._robot.data.root_lin_vel_b * goal_dir_b, dim=1)
        approach_reward_scale = getattr(self.cfg, "approach_reward_scale", 0.0)
        approach_reward = approach_reward_scale * torch.clamp(approach_speed, min=0.0) * self.step_dt
        # 進度獎勵預設沿用「比上一步更近」；特定任務可改成只獎勵刷新本回合最佳距離。
        if bool(getattr(self.cfg, "progress_reward_best_so_far_only", False)):
            progress_to_goal = torch.clamp(self._best_distance_to_goal - distance_to_goal, min=0.0)
        else:
            progress_to_goal = self._prev_distance_to_goal - distance_to_goal
        if bool(getattr(self.cfg, "progress_reward_normalize_by_initial_distance", False)):
            progress_init_dist_min = max(float(getattr(self.cfg, "progress_reward_initial_distance_min", 10.0)), 1e-6)
            progress_den = torch.clamp(self._last_respawn_target_dist, min=progress_init_dist_min)
            progress_to_goal = progress_to_goal / progress_den
        progress_reward_scale = float(getattr(self.cfg, "progress_reward_scale", 0.0))
        progress_reward = progress_reward_scale * progress_to_goal
        speed_to_goal_reward_scale = float(getattr(self.cfg, "speed_to_goal_reward_scale", 0.0))
        speed_to_goal_reward = (
            speed_to_goal_reward_scale
            * torch.clamp(approach_speed, min=0.0)
            * distance_to_goal_mapped
            * self.step_dt
        )
        # r_tcmd = λ4 * ||a_omega,t|| + λ5 * ||a_t - a_{t-1}||^2（在總獎勵中作為懲罰扣除）
        tcmd_lambda_4 = float(getattr(self.cfg, "tcmd_lambda_4", 0.0))
        tcmd_lambda_5 = float(getattr(self.cfg, "tcmd_lambda_5", 0.0))
        a_omega_t = torch.linalg.norm(self._actions[:, 1:], dim=1)
        a_delta_sq = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
        tcmd = (tcmd_lambda_4 * a_omega_t + tcmd_lambda_5 * a_delta_sq) * self.step_dt
        tcmd_penalty = -tcmd

        # 靠近目標時降低速度懲罰，避免策略提早煞車在目標外圍。
        near_touch_scale = torch.clamp(
            distance_to_goal / self.cfg.near_touch_outer_radius,
            min=self.cfg.near_touch_vel_penalty_min_scale,
            max=1.0,
        )

        touch_threshold = self._get_touch_threshold()
        touched = distance_to_goal <= touch_threshold
        near_touch_distance_reward_ratio = float(getattr(self.cfg, "near_touch_distance_reward_ratio", 1.0))
        near_touch_distance_reward_ratio = min(max(near_touch_distance_reward_ratio, 0.0), 1.0)
        near_touch_not_touched = (distance_to_goal < self.cfg.near_touch_outer_radius) & (~touched)
        distance_to_goal_scale = torch.ones_like(distance_to_goal)
        distance_to_goal_scale[near_touch_not_touched] = near_touch_distance_reward_ratio
        distance_to_goal_reward = (
            distance_to_goal_mapped
            * distance_to_goal_scale
            * self.cfg.distance_to_goal_reward_scale
            * self.step_dt
        )
        touch_bonus = (
            self.cfg.touch_bonus_reward * touched.float() if self.cfg.enable_touch_reward else torch.zeros_like(distance_to_goal)
        )
        remaining_frac = 1.0 - (self.episode_length_buf.float() / float(self.max_episode_length))
        touch_early_bonus_scale = getattr(self.cfg, "touch_early_bonus_scale", 0.0)
        touch_early_bonus = (
            touch_early_bonus_scale * remaining_frac * touched.float()
            if self.cfg.enable_touch_reward
            else torch.zeros_like(distance_to_goal)
        )
        time_penalty = -self.cfg.time_penalty_scale * self.step_dt * torch.ones_like(distance_to_goal)
        distance_penalty = -self.cfg.distance_penalty_scale * distance_to_goal * self.step_dt
        if bool(getattr(self.cfg, "distance_penalty_only_when_not_approaching", False)):
            not_approaching = (approach_speed <= 0.0).float()
            distance_penalty = distance_penalty * not_approaching
        lin_speed = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        hovering_near_touch = (
            (distance_to_goal > touch_threshold)
            & (distance_to_goal < self.cfg.near_touch_outer_radius)
            & (lin_speed < self.cfg.near_touch_hover_speed_threshold)
        )
        near_touch_hover_penalty_scale = getattr(self.cfg, "near_touch_hover_penalty", 0.0)
        near_touch_hover_penalty = -near_touch_hover_penalty_scale * hovering_near_touch.float() * self.step_dt
        near_touch_push_reward_scale = float(getattr(self.cfg, "near_touch_push_reward_scale", 0.0))
        near_touch_zone = (
            (distance_to_goal > touch_threshold)
            & (distance_to_goal < self.cfg.near_touch_outer_radius)
            & (approach_speed > 0.0)
        )
        near_touch_zone_span = max(float(self.cfg.near_touch_outer_radius - touch_threshold), 1e-6)
        near_touch_closeness = torch.clamp(
            1.0 - (distance_to_goal - touch_threshold) / near_touch_zone_span,
            min=0.0,
            max=1.0,
        )
        near_touch_push_reward = near_touch_push_reward_scale * near_touch_closeness * near_touch_zone.float() * self.step_dt
        follow_behind_penalty_scale = float(getattr(self.cfg, "follow_behind_penalty_scale", 0.0))
        follow_behind_outer_radius = float(getattr(self.cfg, "follow_behind_outer_radius", self.cfg.near_touch_outer_radius))
        follow_behind_outer_radius = max(follow_behind_outer_radius, touch_threshold + 1e-6)
        follow_behind_min_approach_speed = float(getattr(self.cfg, "follow_behind_min_approach_speed", 0.0))
        follow_behind_zone = (
            (distance_to_goal > touch_threshold)
            & (distance_to_goal < follow_behind_outer_radius)
            & (approach_speed < follow_behind_min_approach_speed)
        )
        follow_behind_closeness = torch.clamp(
            1.0 - (distance_to_goal - touch_threshold) / (follow_behind_outer_radius - touch_threshold),
            min=0.0,
            max=1.0,
        )
        follow_behind_penalty = (
            -follow_behind_penalty_scale
            * follow_behind_closeness
            * follow_behind_zone.float()
            * self.step_dt
        )
        died_by_height = self._get_logical_root_z() < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_by_ground = self._get_ground_contact_mask()
        died_by_tilt = self._get_tilt_exceeded_mask()
        died = died_by_height | died_by_ground | died_by_tilt
        far_away = distance_to_goal > self.cfg.far_away_termination_distance
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        failed_no_touch = time_out & (~touched) & (~died) & (~far_away)
        non_tilt_died = died_by_height | died_by_ground
        death_penalty = -self.cfg.death_penalty * non_tilt_died.float()
        tilt_death_penalty_scale = float(getattr(self.cfg, "tilt_death_penalty", self.cfg.death_penalty))
        tilt_death_penalty = -tilt_death_penalty_scale * died_by_tilt.float()
        timeout_penalty = -self.cfg.death_penalty * time_out.float()
        far_away_penalty_scale = float(getattr(self.cfg, "far_away_penalty", self.cfg.death_penalty))
        far_away_penalty = -far_away_penalty_scale * far_away.float()
        failure_penalty_scale = float(getattr(self.cfg, "failure_penalty", self.cfg.death_penalty))
        failure_penalty = -failure_penalty_scale * failed_no_touch.float()
        tilt_forward_reward_scale = float(getattr(self.cfg, "tilt_forward_reward_scale", 0.0))
        gravity_b = self._robot.data.projected_gravity_b
        g_norm = torch.linalg.norm(gravity_b, dim=1).clamp_min(1e-6)
        cos_tilt = torch.clamp((-gravity_b[:, 2]) / g_norm, -1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
        tilt_min_deg = float(getattr(self.cfg, "tilt_target_min_deg", 30.0))
        tilt_max_deg = float(getattr(self.cfg, "tilt_target_max_deg", 35.0))
        if tilt_min_deg > tilt_max_deg:
            tilt_min_deg, tilt_max_deg = tilt_max_deg, tilt_min_deg
        tilt_sigma_deg = max(float(getattr(self.cfg, "tilt_outside_sigma_deg", 10.0)), 1e-3)
        tilt_below_ratio = float(getattr(self.cfg, "tilt_below_reward_ratio", 0.0))
        tilt_below_ratio = min(max(tilt_below_ratio, 0.0), 1.0)
        in_target_band = (tilt_deg >= tilt_min_deg) & (tilt_deg <= tilt_max_deg)
        below_tilt_deg = torch.clamp(tilt_min_deg - tilt_deg, min=0.0)
        below_tilt_reward = tilt_below_ratio * torch.exp(-torch.square(below_tilt_deg / tilt_sigma_deg))
        over_tilt_deg = torch.clamp(tilt_deg - tilt_max_deg, min=0.0)
        over_tilt_penalty = 1.0 - torch.exp(-torch.square(over_tilt_deg / tilt_sigma_deg))
        moving_toward_goal = (approach_speed > 0.0).float()
        # 軟式傾角 shaping（僅在朝目標前進時啟用）：
        # - 落在目標區間 [tilt_min, tilt_max] 給正獎勵
        # - 超過 tilt_max 給懲罰
        # - 低於 tilt_min 給可調的小幅獎勵
        tilt_forward_reward = (
            tilt_forward_reward_scale
            * (in_target_band.float() + below_tilt_reward - over_tilt_penalty)
            * moving_toward_goal
            * self.step_dt
        )
        tilt_excess_penalty_scale = float(getattr(self.cfg, "tilt_excess_penalty_scale", 0.0))
        tilt_limit_deg = float(getattr(self.cfg, "max_tilt_deg", 35.0))
        tilt_excess_ratio = torch.clamp((tilt_deg - tilt_limit_deg) / max(tilt_limit_deg, 1e-6), min=0.0)
        tilt_excess_penalty = -tilt_excess_penalty_scale * tilt_excess_ratio * self.step_dt
        # 各 reward 組件字典，方便訓練紀錄與除錯。
        rewards = {
            "lin_vel": lin_vel * near_touch_scale * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * near_touch_scale * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_reward,
            "speed_to_goal_reward": speed_to_goal_reward,
            "touch_bonus": touch_bonus,
            "touch_early_bonus": touch_early_bonus,
            "approach_reward": approach_reward,
            "progress_reward": progress_reward,
            "tcmd_penalty": tcmd_penalty,
            "time_penalty": time_penalty,
            "timeout_penalty": timeout_penalty,
            "near_touch_hover_penalty": near_touch_hover_penalty,
            "near_touch_push_reward": near_touch_push_reward,
            "follow_behind_penalty": follow_behind_penalty,
            "distance_penalty": distance_penalty,
            "death_penalty": death_penalty,
            "tilt_death_penalty": tilt_death_penalty,
            "tilt_excess_penalty": tilt_excess_penalty,
            "tilt_forward_reward": tilt_forward_reward,
            "far_away_penalty": far_away_penalty,
            "failure_penalty": failure_penalty,
        }
        # 總獎勵 = 各項 shaping 與終局導向項的加總。
        reward = torch.sum(
            torch.stack([value for key, value in rewards.items() if key != "timeout_penalty"]),
            dim=0,
        )

        self._prev_distance_to_goal = distance_to_goal.detach()
        self._best_distance_to_goal = torch.minimum(self._best_distance_to_goal, distance_to_goal.detach())
        self._prev_actions.copy_(self._actions.detach())
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳 (terminated, time_out)。

        terminated：死亡/太遠/碰觸（若設定啟用）
        time_out：回合步數達上限
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        died_by_height = self._get_logical_root_z() < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_by_tilt = self._get_tilt_exceeded_mask()
        died = died_by_height | self._get_ground_contact_mask() | died_by_tilt
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        far_away = distance_to_goal > self.cfg.far_away_termination_distance
        touched = distance_to_goal <= self._get_touch_threshold()
        self._term_touched = touched
        terminated = died | far_away | (touched if self.cfg.terminate_on_touch else torch.zeros_like(touched))
        return terminated, time_out

    def _reset_idx_impl(self, env_ids: torch.Tensor | None, spread_episode_resets: bool):
        """實際重置流程（訓練/測試共用核心）。

        包含：
        - 回合統計寫入 log
        - 機體狀態重置與重生採樣
        - 目標點重生（含可選距離課程）
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        distance_to_goal_all = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        touched_mask = distance_to_goal_all <= self._get_touch_threshold()
        far_away_mask = distance_to_goal_all > self.cfg.far_away_termination_distance
        died_by_height_mask = self._get_logical_root_z(env_ids) < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_by_tilt_mask = self._get_tilt_exceeded_mask(env_ids)
        died_mask = died_by_height_mask | self._get_ground_contact_mask(env_ids) | died_by_tilt_mask
        desired_pos_b_env, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w[env_ids],
            self._robot.data.root_quat_w[env_ids],
            self._desired_pos_w[env_ids],
        )
        goal_dir_b_env = desired_pos_b_env / (torch.linalg.norm(desired_pos_b_env, dim=1, keepdim=True) + 1e-6)
        approach_speed_env = torch.sum(self._robot.data.root_lin_vel_b[env_ids] * goal_dir_b_env, dim=1)
        approaching_mask = approach_speed_env > 0.0

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            # 僅保留 raw reward 統計，避免重複輸出正規化版 Episode_Reward。
            extras["Episode_RewardRaw/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"].update(
            {
                "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
                "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
                "Episode_Termination/touched": torch.count_nonzero(self._term_touched[env_ids]).item(),
                "Episode_Termination/failed_no_touch": torch.count_nonzero(
                    self.reset_time_outs[env_ids] & (~self._term_touched[env_ids])
                ).item(),
                "Metrics/final_distance_to_goal": final_distance_to_goal.item(),
                "Metrics/approaching_rate": torch.mean(approaching_mask.float()).item(),
                "Metrics/touched_rate": torch.mean(touched_mask.float()).item(),
                "Metrics/far_away_rate": torch.mean(far_away_mask.float()).item(),
                "Metrics/died_rate": torch.mean(died_mask.float()).item(),
            }
        )

        # 先重置機體，再寫入新的 root/joint 狀態。
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if spread_episode_resets and len(env_ids) == self.num_envs:
            # 將整批 reset 的 episode 長度打散，避免同時重置造成吞吐尖峰。
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        # 目標高度範圍預設沿用低空設定；展示環境可把它整段抬到建物上空。
        target_spawn_z_min = float(getattr(self.cfg, "target_spawn_z_min", 0.5))
        target_spawn_z_max = float(getattr(self.cfg, "target_spawn_z_max", 5.0))
        if target_spawn_z_min > target_spawn_z_max:
            target_spawn_z_min, target_spawn_z_max = target_spawn_z_max, target_spawn_z_min

        # 先做預設目標取樣；若啟用 target distance 規則，後面會覆寫 XY。
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-5.0, 5.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(
            target_spawn_z_min, target_spawn_z_max
        )

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        spawn_z_max = self.cfg.spawn_z_max
        spawn_xy_min = self.cfg.spawn_xy_min
        spawn_xy_max = self.cfg.spawn_xy_max
        target_dist_curriculum_enabled = bool(getattr(self.cfg, "target_distance_curriculum_enabled", False))
        if getattr(self.cfg, "curriculum_enabled", False):
            self._curriculum_step += int(len(env_ids))
        if target_dist_curriculum_enabled:
            self._target_dist_curriculum_step += 1
        # 機體重生範圍課程（spawn z/xy）。
        if getattr(self.cfg, "curriculum_enabled", False):
            ramp = max(1, int(getattr(self.cfg, "curriculum_ramp_steps", 1)))
            frac = min(float(self._curriculum_step) / float(ramp), 1.0)
            if hasattr(self.cfg, "curriculum_spawn_z_max_start") and hasattr(self.cfg, "curriculum_spawn_z_max_end"):
                start = float(getattr(self.cfg, "curriculum_spawn_z_max_start"))
                end = float(getattr(self.cfg, "curriculum_spawn_z_max_end"))
                spawn_z_max = max(float(self.cfg.spawn_z_min), start + (end - start) * frac)
            else:
                stages = list(getattr(self.cfg, "curriculum_spawn_max_stages", (spawn_z_max,)))
                stage_idx = min(int(frac * len(stages)), len(stages) - 1)
                if not hasattr(self, "_curriculum_stage_idx"):
                    self._curriculum_stage_idx = -1
                if stage_idx != self._curriculum_stage_idx:
                    self._curriculum_stage_idx = stage_idx
                    stage_max = float(stages[stage_idx])
                    print(
                        f"[CURRICULUM][Touch] stage={stage_idx + 1}/{len(stages)} "
                        f"spawn_range=1~{stage_max}m",
                        flush=True,
                    )
                spawn_z_max = max(float(stages[stage_idx]), float(self.cfg.spawn_z_min))
                spawn_xy_min = -float(stages[stage_idx])
                spawn_xy_max = float(stages[stage_idx])
        spawn_xy = torch.empty(
            (len(env_ids), 2),
            device=default_root_state.device,
            dtype=default_root_state.dtype,
        ).uniform_(spawn_xy_min, spawn_xy_max)
        spawn_z = torch.empty(
            (len(env_ids),),
            device=default_root_state.device,
            dtype=default_root_state.dtype,
        ).uniform_(self.cfg.spawn_z_min, spawn_z_max)
        height_offset = self._get_visual_altitude_offset()
        if bool(getattr(self.cfg, "scene_anchor_enabled", False)) and self._scene_anchor_centers_w is not None:
            anchor_spawn_xy = self._sample_scene_anchor_spawn_xy(env_ids, dtype=default_root_state.dtype)
            default_root_state[:, 0] = anchor_spawn_xy[:, 0]
            default_root_state[:, 1] = anchor_spawn_xy[:, 1]
        else:
            default_root_state[:, 0] = self._terrain.env_origins[env_ids, 0] + spawn_xy[:, 0]
            default_root_state[:, 1] = self._terrain.env_origins[env_ids, 1] + spawn_xy[:, 1]
        default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + spawn_z + height_offset
        # 展示模式可把觀測原點固定在本回合重生點，避免把公里級絕對 x/y 餵給 Stage4 policy。
        self._observation_local_origin_xy[env_ids] = default_root_state[:, :2]
        target_dist_min = getattr(self.cfg, "target_spawn_distance_min", None)
        target_dist_max = getattr(self.cfg, "target_spawn_distance_max", None)
        if target_dist_min is not None and target_dist_max is not None:
            min_dist = float(target_dist_min)
            max_dist = float(target_dist_max)
            if target_dist_curriculum_enabled:
                # 目標距離課程：可獨立於 spawn curriculum 啟用。
                stages = getattr(self.cfg, "target_distance_curriculum_stages", None)
                dist_ramp = int(getattr(self.cfg, "target_distance_curriculum_ramp_steps", getattr(self.cfg, "curriculum_ramp_steps", 1)))
                dist_ramp = max(1, dist_ramp)
                raw_dist_frac = min(float(self._target_dist_curriculum_step) / float(dist_ramp), 1.0)
                progress_power = float(getattr(self.cfg, "target_distance_curriculum_progress_power", 1.0))
                progress_power = max(progress_power, 1e-6)
                dist_frac = min(max(raw_dist_frac**progress_power, 0.0), 1.0)
                if stages:
                    stage_pairs: list[tuple[float, float]] = [(float(s[0]), float(s[1])) for s in stages]
                    stage_count = len(stage_pairs)
                    stage_steps_cfg = getattr(self.cfg, "target_distance_curriculum_stage_steps", None)
                    stage_idx = None
                    if stage_steps_cfg is not None:
                        stage_steps = [max(1, int(x)) for x in stage_steps_cfg]
                        if len(stage_steps) == stage_count:
                            progress_step = int(getattr(self, "common_step_counter", self._target_dist_curriculum_step))
                            accum_steps = 0
                            for i, stage_steps_i in enumerate(stage_steps):
                                accum_steps += stage_steps_i
                                if progress_step < accum_steps:
                                    stage_idx = i
                                    break
                            if stage_idx is None:
                                stage_idx = stage_count - 1
                        else:
                            stage_idx = min(int(dist_frac * stage_count), stage_count - 1)
                    else:
                        stage_idx = min(int(dist_frac * stage_count), stage_count - 1)
                    min_dist, max_dist = stage_pairs[stage_idx]
                    if min_dist > max_dist:
                        min_dist, max_dist = max_dist, min_dist
                    if not hasattr(self, "_target_dist_curriculum_stage_idx"):
                        self._target_dist_curriculum_stage_idx = -1
                    if stage_idx != self._target_dist_curriculum_stage_idx:
                        self._target_dist_curriculum_stage_idx = stage_idx
                        print(
                            f"[CURRICULUM][TargetDist] stage={stage_idx + 1}/{stage_count} "
                            f"range={min_dist:.1f}~{max_dist:.1f}m",
                            flush=True,
                        )
                else:
                    min_start = getattr(self.cfg, "target_distance_curriculum_min_start", None)
                    min_end = getattr(self.cfg, "target_distance_curriculum_min_end", None)
                    max_start = getattr(self.cfg, "target_distance_curriculum_max_start", None)
                    max_end = getattr(self.cfg, "target_distance_curriculum_max_end", None)
                    if min_start is None:
                        min_start = getattr(self.cfg, "target_distance_curriculum_start", min_dist)
                    if max_end is None:
                        max_end = getattr(self.cfg, "target_distance_curriculum_end", max_dist)
                    if min_end is None:
                        min_end = min_dist
                    if max_start is None:
                        max_start = float(min_start)
                    min_start = float(min_start)
                    min_end = float(min_end)
                    max_start = float(max_start)
                    max_end = float(max_end)
                    min_dist = min_start + (min_end - min_start) * dist_frac
                    max_dist = max_start + (max_end - max_start) * dist_frac
                    print_stages = max(1, int(getattr(self.cfg, "target_distance_curriculum_print_stages", 10)))
                    stage_idx = min(int(dist_frac * print_stages), print_stages - 1)
                    if not hasattr(self, "_target_dist_curriculum_stage_idx"):
                        self._target_dist_curriculum_stage_idx = -1
                    if stage_idx != self._target_dist_curriculum_stage_idx:
                        self._target_dist_curriculum_stage_idx = stage_idx
                        print(
                            f"[CURRICULUM][TargetDist] stage={stage_idx + 1}/{print_stages} "
                            f"range={min_dist:.1f}~{max_dist:.1f}m",
                            flush=True,
                        )
            if min_dist > max_dist:
                min_dist, max_dist = max_dist, min_dist
            # 以無人機重生點為圓心，在指定平面距離範圍內隨機重生目標。
            target_theta = torch.empty((len(env_ids),), device=default_root_state.device, dtype=default_root_state.dtype).uniform_(
                0.0, 2.0 * torch.pi
            )
            target_radius = torch.empty(
                (len(env_ids),), device=default_root_state.device, dtype=default_root_state.dtype
            ).uniform_(min_dist, max_dist)
            self._desired_pos_w[env_ids, 0] = default_root_state[:, 0] + target_radius * torch.cos(target_theta)
            self._desired_pos_w[env_ids, 1] = default_root_state[:, 1] + target_radius * torch.sin(target_theta)
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(
                target_spawn_z_min, target_spawn_z_max
            )
            self._desired_pos_w[env_ids, 2] += self._terrain.env_origins[env_ids, 2] + height_offset
            if bool(getattr(self.cfg, "scene_anchor_enabled", False)):
                target_extra_clearance = float(getattr(self.cfg, "scene_anchor_target_extra_clearance_m", 0.0))
                self._desired_pos_w[env_ids] = self._enforce_scene_anchor_clearance(
                    self._desired_pos_w[env_ids], env_ids, target_extra_clearance
                )
                self._last_respawn_target_dist[env_ids] = torch.linalg.norm(
                    self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
                )
            else:
                self._last_respawn_target_dist[env_ids] = target_radius
        else:
            self._desired_pos_w[env_ids, 2] += self._terrain.env_origins[env_ids, 2] + height_offset
            if bool(getattr(self.cfg, "scene_anchor_enabled", False)):
                self._desired_pos_w[env_ids] = self._enforce_scene_anchor_clearance(
                    self._desired_pos_w[env_ids], env_ids
                )
            self._last_respawn_target_dist[env_ids] = torch.linalg.norm(
                self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
            )
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._prev_distance_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )
        self._best_distance_to_goal[env_ids] = self._prev_distance_to_goal[env_ids]
        self._term_touched[env_ids] = False

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """預設 reset 入口（訓練模式使用，含重置攤平）。"""
        self._reset_idx_impl(env_ids, spread_episode_resets=True)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """建立/切換目標可視化標記。"""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                # 若 cfg 指定固定邊長，則方塊顯示和實際 threshold 脫鉤；否則沿用 threshold 顯示。
                marker_edge_length = getattr(self.cfg, "touch_marker_edge_length", None)
                if marker_edge_length is None:
                    touch_diameter = self._get_touch_threshold() * 2.0
                    marker_scale = 0.9
                    marker_size = touch_diameter * marker_scale
                else:
                    marker_size = float(marker_edge_length)
                marker_cfg.markers["cuboid"].size = (marker_size, marker_size, marker_size)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """每個 debug callback 更新目標標記位置。"""
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._desired_pos_w)


class DroneTargetTouchTestEnv(DroneTargetTouchEnv):
    """測試環境：沿用訓練環境行為，並輸出可比較的統計資訊。"""

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """測試重置入口：輸出成功率/步數統計，並使用可重現重置策略。"""
        if not hasattr(self, "_test_reset_counter"):
            self._test_reset_counter = 0
            self._test_total_episodes = 0
            self._test_total_success = 0
            self._test_total_success_steps = 0.0
            print("[TEST][Touch] DroneTargetTouchTestEnv reset hook active", flush=True)
        self._test_reset_counter += 1
        # 測試模式：關閉課程，讓重生高度直接使用完整範圍。
        self.cfg.curriculum_enabled = False
        # Touch 測試：改用固定目標距離範圍，不走距離課程。
        if getattr(self.cfg, "target_distance_curriculum_enabled", False):
            self.cfg.target_distance_curriculum_enabled = False
            self.cfg.target_spawn_distance_min = 20.0
            self.cfg.target_spawn_distance_max = 50.0
        # 測試模式：降低高度死亡門檻，減少近地面過早終止。
        self.cfg.died_height_threshold = 0.1

        if env_ids is None or len(env_ids) == self.num_envs:
            log_env_ids = self._robot._ALL_INDICES
        else:
            log_env_ids = env_ids

        touched_mask = self._term_touched[log_env_ids]
        touched_count = int(touched_mask.sum().item())
        batch_episodes = int(touched_mask.numel())
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[log_env_ids] - self._robot.data.root_pos_w[log_env_ids], dim=1
        )
        died_by_height = self._get_logical_root_z(log_env_ids) < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_by_ground = self._get_ground_contact_mask(log_env_ids)
        died_by_tilt = self._get_tilt_exceeded_mask(log_env_ids)
        died_mask = died_by_height | died_by_ground | died_by_tilt
        far_mask = distance_to_goal > self.cfg.far_away_termination_distance
        time_out_mask = self.reset_time_outs[log_env_ids]
        fail_no_touch_mask = time_out_mask & (~touched_mask) & (~died_mask) & (~far_mask)
        reason_summary = (
            f"dh:{int(died_by_height.sum().item())},"
            f"dg:{int(died_by_ground.sum().item())},"
            f"dt:{int(died_by_tilt.sum().item())},"
            f"fa:{int(far_mask.sum().item())},"
            f"to:{int(time_out_mask.sum().item())},"
            f"fn:{int(fail_no_touch_mask.sum().item())}"
        )
        self._test_total_episodes += batch_episodes
        self._test_total_success += touched_count
        total_success_rate = (self._test_total_success / self._test_total_episodes) if self._test_total_episodes > 0 else 0.0
        batch_total_reward = torch.zeros(len(log_env_ids), dtype=torch.float, device=self.device)
        for key in self._episode_sums.keys():
            batch_total_reward += self._episode_sums[key][log_env_ids]
        reward_summary = (
            f"reward={float(batch_total_reward.mean().item()):.2f}"
            f"(min:{float(batch_total_reward.min().item()):.2f},"
            f"max:{float(batch_total_reward.max().item()):.2f})"
        )
        # 測試模式使用可重現的重置節奏（不打散 episode 長度）。
        self._reset_idx_impl(env_ids, spread_episode_resets=False)
        respawn_dist = self._last_respawn_target_dist[log_env_ids]
        respawn_dist_summary = (
            f"rd={float(respawn_dist.mean().item()):.1f}"
            f"(min:{float(respawn_dist.min().item()):.1f},"
            f"max:{float(respawn_dist.max().item()):.1f})"
        )
        if touched_count > 0:
            touched_steps = self.episode_length_buf[log_env_ids][touched_mask].float()
            self._test_total_success_steps += float(touched_steps.sum().item())
            total_avg_steps = self._test_total_success_steps / max(self._test_total_success, 1)
            print(
                "[TEST][Touch] count="
                f"{touched_count}, "
                f"max_steps={float(touched_steps.max().item()):.1f}, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_steps={total_avg_steps:.1f}, "
                f"{reward_summary}, "
                f"{respawn_dist_summary}, "
                f"rsn={reason_summary}"
                ,
                flush=True,
            )
        else:
            print(
                "[TEST][Touch] count=0, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_steps={self._test_total_success_steps / max(self._test_total_success, 1):.1f}, "
                f"{reward_summary}, "
                f"{respawn_dist_summary}, "
                f"rsn={reason_summary}",
                flush=True,
            )
