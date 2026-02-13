# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms

from .drone_env_target_touch_cfg import DroneTargetTouchEnvCfg
from .markers import CUBOID_MARKER_CFG, VisualizationMarkers


class DroneTargetTouchEnv(DirectRLEnv):
    """Hover-aligned baseline environment (same core setup as Hover)."""

    cfg: DroneTargetTouchEnvCfg

    def __init__(self, cfg: DroneTargetTouchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "touch_bonus",
                "touch_early_bonus",
                "approach_reward",
                "time_penalty",
                "near_touch_hover_penalty",
                "distance_penalty",
                "death_penalty",
                "far_away_penalty",
                "failure_penalty",
            ]
        }
        self._term_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._curriculum_step = 0

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
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
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
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
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1.0 - torch.tanh(distance_to_goal / 0.8)
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        goal_dir_b = desired_pos_b / (torch.linalg.norm(desired_pos_b, dim=1, keepdim=True) + 1e-6)
        approach_speed = torch.sum(self._robot.data.root_lin_vel_b * goal_dir_b, dim=1)
        approach_reward_scale = getattr(self.cfg, "approach_reward_scale", 0.0)
        approach_reward = approach_reward_scale * torch.clamp(approach_speed, min=0.0) * self.step_dt

        # 靠近目標時降低速度懲罰，避免策略提早煞車在目標外圍。
        near_touch_scale = torch.clamp(
            distance_to_goal / self.cfg.near_touch_outer_radius,
            min=self.cfg.near_touch_vel_penalty_min_scale,
            max=1.0,
        )

        touched = distance_to_goal <= self.cfg.touch_radius
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
        lin_speed = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        hovering_near_touch = (
            (distance_to_goal > self.cfg.touch_radius)
            & (distance_to_goal < self.cfg.near_touch_outer_radius)
            & (lin_speed < self.cfg.near_touch_hover_speed_threshold)
        )
        near_touch_hover_penalty_scale = getattr(self.cfg, "near_touch_hover_penalty", 0.0)
        near_touch_hover_penalty = -near_touch_hover_penalty_scale * hovering_near_touch.float() * self.step_dt
        died = self._robot.data.root_pos_w[:, 2] < 0.3
        far_away = distance_to_goal > self.cfg.far_away_termination_distance
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        failed_no_touch = time_out & (~touched) & (~died) & (~far_away)
        death_penalty = -self.cfg.death_penalty * died.float()
        far_away_penalty = -self.cfg.death_penalty * far_away.float()
        failure_penalty = -self.cfg.death_penalty * failed_no_touch.float()
        rewards = {
            "lin_vel": lin_vel * near_touch_scale * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * near_touch_scale * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "touch_bonus": touch_bonus,
            "touch_early_bonus": touch_early_bonus,
            "approach_reward": approach_reward,
            "time_penalty": time_penalty,
            "near_touch_hover_penalty": near_touch_hover_penalty,
            "distance_penalty": distance_penalty,
            "death_penalty": death_penalty,
            "far_away_penalty": far_away_penalty,
            "failure_penalty": failure_penalty,
        }
        # 總獎勵 = 各項 shaping 與終局導向項的加總。
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self._robot.data.root_pos_w[:, 2] < 0.3
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        far_away = distance_to_goal > self.cfg.far_away_termination_distance
        touched = distance_to_goal <= self.cfg.touch_radius
        self._term_touched = touched
        terminated = died | far_away | (touched if self.cfg.terminate_on_touch else torch.zeros_like(touched))
        return terminated, time_out

    def _reset_idx_impl(self, env_ids: torch.Tensor | None, spread_episode_resets: bool):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
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
            }
        )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if spread_episode_resets and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes when many environments reset at a similar time.
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # target sampling (same as Hover baseline)
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
        if getattr(self.cfg, "curriculum_enabled", False):
            self._curriculum_step += int(len(env_ids))
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
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._term_touched[env_ids] = False

    def _reset_idx(self, env_ids: torch.Tensor | None):
        self._reset_idx_impl(env_ids, spread_episode_resets=True)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                # Visualize the touch radius as a cube with side length = 2 * radius.
                touch_diameter = float(self.cfg.touch_radius) * 2.0
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
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._desired_pos_w)


class DroneTargetTouchTestEnv(DroneTargetTouchEnv):
    """Use same behavior as training env for baseline comparison."""

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if not hasattr(self, "_test_reset_counter"):
            self._test_reset_counter = 0
            self._test_total_episodes = 0
            self._test_total_success = 0
            self._test_total_success_time_s = 0.0
            print("[TEST][Touch] DroneTargetTouchTestEnv reset hook active", flush=True)
        self._test_reset_counter += 1
        # Test env: disable curriculum so spawn height uses full range immediately.
        self.cfg.curriculum_enabled = False

        if env_ids is None or len(env_ids) == self.num_envs:
            log_env_ids = self._robot._ALL_INDICES
        else:
            log_env_ids = env_ids

        touched_mask = self._term_touched[log_env_ids]
        touched_count = int(touched_mask.sum().item())
        batch_episodes = int(touched_mask.numel())
        self._test_total_episodes += batch_episodes
        self._test_total_success += touched_count
        batch_success_rate = (touched_count / batch_episodes) if batch_episodes > 0 else 0.0
        total_success_rate = (self._test_total_success / self._test_total_episodes) if self._test_total_episodes > 0 else 0.0
        if touched_count > 0:
            touched_steps = self.episode_length_buf[log_env_ids][touched_mask].float()
            touched_time_s = touched_steps * self.step_dt
            self._test_total_success_time_s += float(touched_time_s.sum().item())
            total_avg_time = self._test_total_success_time_s / max(self._test_total_success, 1)
            print(
                "[TEST][Touch] count="
                f"{touched_count}, "
                f"avg_time={float(touched_time_s.mean().item()):.3f}s, "
                f"max_time={float(touched_time_s.max().item()):.3f}s, "
                f"rate={batch_success_rate:.3f}, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_time={total_avg_time:.3f}s"
                ,
                flush=True,
            )
        else:
            print(
                "[TEST][Touch] count=0, "
                f"rate={batch_success_rate:.3f}, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_time={self._test_total_success_time_s / max(self._test_total_success, 1):.3f}s",
                flush=True,
            )

        # Keep all reset behavior identical to DroneTargetTouchEnv,
        # including random episode-length staggering.
        self._reset_idx_impl(env_ids, spread_episode_resets=True)
