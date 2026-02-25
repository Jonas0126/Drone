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
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

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

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_respawn_target_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._prev_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        reward_log_keys = [
            "lin_vel",
            "ang_vel",
            "distance_to_goal",
            "touch_bonus",
            "touch_early_bonus",
            "approach_reward",
            "tcmd_penalty",
            "time_penalty",
            "near_touch_hover_penalty",
            "distance_penalty",
            "death_penalty",
            "tilt_forward_reward",
            "far_away_penalty",
            "failure_penalty",
        ]
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_log_keys
        }
        self._term_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._curriculum_step = 0
        # 目標距離課程專用步數計數器：避免大量環境同時 reset 時課程升級過快。
        self._target_dist_curriculum_step = 0
        self._drone_body_sphere_radius = self._resolve_drone_body_sphere_radius()

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
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
            ground_z = self._terrain.env_origins[:, 2]
        else:
            root_z = self._robot.data.root_pos_w[env_ids, 2]
            ground_z = self._terrain.env_origins[env_ids, 2]

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

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
        - touch_bonus: 進入 touch 半徑時的一次性正獎勵。
        - touch_early_bonus: 越早碰到目標，額外加分越多。
        - approach_reward: 在機體座標系中，朝目標方向前進速度的正獎勵。
        - time_penalty: 每一步固定扣分，鼓勵更短時間完成。
        - near_touch_hover_penalty: 靠近目標卻低速懸停、未實際碰觸時的懲罰。
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
        died_by_height = self._robot.data.root_pos_w[:, 2] < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died = died_by_height | self._get_ground_contact_mask()
        far_away = distance_to_goal > self.cfg.far_away_termination_distance
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        failed_no_touch = time_out & (~touched) & (~died) & (~far_away)
        death_penalty = -self.cfg.death_penalty * died.float()
        far_away_penalty = -self.cfg.death_penalty * far_away.float()
        failure_penalty = -self.cfg.death_penalty * failed_no_touch.float()
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
        # 各 reward 組件字典，方便訓練紀錄與除錯。
        rewards = {
            "lin_vel": lin_vel * near_touch_scale * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * near_touch_scale * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "touch_bonus": touch_bonus,
            "touch_early_bonus": touch_early_bonus,
            "approach_reward": approach_reward,
            "tcmd_penalty": tcmd_penalty,
            "time_penalty": time_penalty,
            "near_touch_hover_penalty": near_touch_hover_penalty,
            "distance_penalty": distance_penalty,
            "death_penalty": death_penalty,
            "tilt_forward_reward": tilt_forward_reward,
            "far_away_penalty": far_away_penalty,
            "failure_penalty": failure_penalty,
        }
        # 總獎勵 = 各項 shaping 與終局導向項的加總。
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        self._prev_distance_to_goal = distance_to_goal.detach()
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

        died_by_height = self._robot.data.root_pos_w[:, 2] < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died = died_by_height | self._get_ground_contact_mask()
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
        died_by_height_mask = self._robot.data.root_pos_w[env_ids, 2] < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_mask = died_by_height_mask | self._get_ground_contact_mask(env_ids)
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
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
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

        # 先做預設目標取樣；若啟用 target distance 規則，後面會覆寫 XY。
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-5.0, 5.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 5.0)

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
        default_root_state[:, 0] = self._terrain.env_origins[env_ids, 0] + spawn_xy[:, 0]
        default_root_state[:, 1] = self._terrain.env_origins[env_ids, 1] + spawn_xy[:, 1]
        default_root_state[:, 2] = spawn_z
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
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 5.0)
            self._last_respawn_target_dist[env_ids] = target_radius
        else:
            self._last_respawn_target_dist[env_ids] = torch.linalg.norm(
                self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
            )
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._prev_distance_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )
        self._term_touched[env_ids] = False

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """預設 reset 入口（訓練模式使用，含重置攤平）。"""
        self._reset_idx_impl(env_ids, spread_episode_resets=True)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """建立/切換目標可視化標記。"""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                # 以立方體顯示觸碰門檻（邊長 = 2 * threshold）。
                touch_diameter = self._get_touch_threshold() * 2.0
                marker_scale = 0.9
                marker_size = touch_diameter * marker_scale
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
        died_by_height = self._robot.data.root_pos_w[log_env_ids, 2] < float(getattr(self.cfg, "died_height_threshold", 0.3))
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
