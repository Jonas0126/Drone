# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .cfg import DroneTargetTouchEnvCfg
from .debug_vis import debug_vis_callback, set_debug_vis
from .observations import build_policy_observations
from .reset_ops import reset_idx_impl, reset_idx_test, reset_idx_train
from .rewards import compute_rewards
from .scene_ops import setup_scene
from .terminations import compute_dones


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
        setup_scene(self)

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
        return build_policy_observations(self)

    def _get_rewards(self) -> torch.Tensor:
        """計算目標碰觸任務獎勵（Moving 版本也沿用本函式）。"""
        return compute_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳 (terminated, time_out)。"""
        return compute_dones(self)

    def _reset_idx_impl(self, env_ids: torch.Tensor | None, spread_episode_resets: bool):
        """實際重置流程（訓練/測試共用核心）。"""
        reset_idx_impl(self, env_ids, spread_episode_resets, self._call_base_reset_idx)

    def _call_base_reset_idx(self, env_ids: torch.Tensor) -> None:
        """Call DirectRLEnv's reset path from split reset helpers."""
        super()._reset_idx(env_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """預設 reset 入口（訓練模式使用，含重置攤平）。"""
        reset_idx_train(self, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """建立/切換目標可視化標記。"""
        set_debug_vis(self, debug_vis)

    def _debug_vis_callback(self, event):
        """每個 debug callback 更新目標標記位置。"""
        debug_vis_callback(self, event)


class DroneTargetTouchTestEnv(DroneTargetTouchEnv):
    """測試環境：沿用訓練環境行為，並輸出可比較的統計資訊。"""

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """測試重置入口：輸出成功率/步數統計，並使用可重現重置策略。"""
        reset_idx_test(self, env_ids)
