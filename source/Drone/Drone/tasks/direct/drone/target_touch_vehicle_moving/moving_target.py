from __future__ import annotations

import torch

from .motion_sampling import (
    sample_heading_hold_steps,
    sample_target_speeds,
    sample_turn_decision_steps,
    sample_turn_segment_steps,
)
from .obstacle_ops import apply_scene_obstacle_avoidance, push_targets_out_of_scene_obstacles


def init_moving_target_state(env) -> None:
    """Initialize per-env moving-target state buffers."""
    env._target_velocity_w = torch.zeros(env.num_envs, 3, device=env.device)
    env._target_dir_w = torch.zeros(env.num_envs, 3, device=env.device)
    env._target_base_speed_mps = torch.full((env.num_envs,), float(env.cfg.moving_target_speed), device=env.device)
    env._target_speed_mps = torch.full((env.num_envs,), float(env.cfg.moving_target_speed), device=env.device)
    env._target_heading_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._target_yaw_w = torch.zeros(env.num_envs, device=env.device)
    env._target_turn_rate_rad_s = torch.zeros(env.num_envs, device=env.device)
    env._target_turn_decision_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._target_z_phase = torch.zeros(env.num_envs, device=env.device)
    env._episode_start_distance = torch.full((env.num_envs,), float("nan"), device=env.device)
    env._episode_min_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
    env._episode_min_height = torch.full((env.num_envs,), float("inf"), device=env.device)
    env._episode_speed_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    env._episode_speed_steps = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    env._episode_speed_max = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    env._moving_target_dist_curriculum_stage_idx = -1


def update_moving_targets(env, env_ids: torch.Tensor) -> None:
    prev_dir_w = env._target_dir_w[env_ids]
    prev_norm = torch.linalg.norm(prev_dir_w, dim=1, keepdim=True)
    first_update = prev_norm.squeeze(-1) < 1e-6
    safe_prev_dir_w = prev_dir_w / torch.clamp(prev_norm, min=1e-6)
    motion_mode = str(getattr(env.cfg, "moving_target_motion_mode", "evade_drone")).lower()

    if motion_mode == "road_like":
        decision_steps = env._target_turn_decision_steps[env_ids] - 1
        env._target_turn_decision_steps[env_ids] = decision_steps
        resample_mask = first_update | (decision_steps <= 0)
        if torch.any(resample_mask):
            resample_env_ids = env_ids[resample_mask]
            sample_count = int(resample_mask.sum().item())
            straight_prob = min(max(float(getattr(env.cfg, "moving_target_straight_prob", 0.6)), 0.0), 1.0)
            is_straight = torch.rand(sample_count, device=env.device) < straight_prob
            turn_rates = torch.zeros(sample_count, device=env.device)
            min_turn_rate = max(float(getattr(env.cfg, "moving_target_turn_rate_min_deg_s", 8.0)), 0.0)
            max_turn_rate = max(float(getattr(env.cfg, "moving_target_turn_rate_max_deg_s", min_turn_rate)), min_turn_rate)
            turning_mask = ~is_straight
            if torch.any(turning_mask):
                turn_count = int(turning_mask.sum().item())
                mags_deg = torch.empty(turn_count, device=env.device).uniform_(min_turn_rate, max_turn_rate)
                mags_rad = torch.deg2rad(mags_deg)
                signs = torch.where(
                    torch.rand(turn_count, device=env.device) < 0.5,
                    -torch.ones(turn_count, device=env.device),
                    torch.ones(turn_count, device=env.device),
                )
                turn_rates[turning_mask] = signs * mags_rad
            env._target_turn_rate_rad_s[resample_env_ids] = turn_rates
            decision_steps = sample_turn_decision_steps(env, sample_count)
            if torch.any(turning_mask):
                decision_steps[turning_mask] = sample_turn_segment_steps(env, int(turning_mask.sum().item()))
            env._target_turn_decision_steps[resample_env_ids] = decision_steps
            turn_speed_ratio = float(getattr(env.cfg, "moving_target_turn_speed_ratio", 0.75))
            turn_speed_ratio = max(turn_speed_ratio, 0.0)
            env._target_speed_mps[resample_env_ids] = env._target_base_speed_mps[resample_env_ids]
            if torch.any(turning_mask):
                turning_env_ids = resample_env_ids[turning_mask]
                env._target_speed_mps[turning_env_ids] = env._target_base_speed_mps[turning_env_ids] * turn_speed_ratio

        env._target_yaw_w[env_ids] = env._target_yaw_w[env_ids] + env._target_turn_rate_rad_s[env_ids] * env.step_dt
        new_dir_w = torch.zeros(len(env_ids), 3, device=env.device)
        new_dir_w[:, 0] = torch.cos(env._target_yaw_w[env_ids])
        new_dir_w[:, 1] = torch.sin(env._target_yaw_w[env_ids])
    else:
        if motion_mode == "random_straight":
            desired_dir_w = safe_prev_dir_w.clone()
            hold_steps = env._target_heading_hold_steps[env_ids] - 1
            env._target_heading_hold_steps[env_ids] = hold_steps
            resample_mask = first_update | (hold_steps <= 0)
            if torch.any(resample_mask):
                resample_env_ids = env_ids[resample_mask]
                sample_count = int(resample_mask.sum().item())
                random_dir_w = torch.zeros(sample_count, 3, device=env.device)
                min_turn_deg = float(getattr(env.cfg, "moving_target_heading_min_turn_deg", 0.0))
                max_turn_deg = float(getattr(env.cfg, "moving_target_heading_max_turn_deg", 180.0))
                min_turn_deg = max(0.0, min(180.0, min_turn_deg))
                max_turn_deg = max(0.0, min(180.0, max_turn_deg))
                if min_turn_deg > max_turn_deg:
                    min_turn_deg, max_turn_deg = max_turn_deg, min_turn_deg
                min_turn_rad = torch.deg2rad(torch.tensor(min_turn_deg, device=env.device))
                max_turn_rad = torch.deg2rad(torch.tensor(max_turn_deg, device=env.device))
                prev_sel = safe_prev_dir_w[resample_mask]
                prev_xy = prev_sel[:, :2]
                prev_xy_norm = torch.linalg.norm(prev_xy, dim=1, keepdim=True)
                prev_valid = prev_xy_norm.squeeze(-1) > 1e-6
                prev_xy_unit = prev_xy / torch.clamp(prev_xy_norm, min=1e-6)
                base_yaw = torch.atan2(prev_xy_unit[:, 1], prev_xy_unit[:, 0])
                if max_turn_rad <= 0.0:
                    delta_yaw = torch.zeros(sample_count, device=env.device)
                elif min_turn_rad <= 0.0:
                    delta_yaw = (torch.rand(sample_count, device=env.device) * 2.0 - 1.0) * max_turn_rad
                else:
                    signs = torch.where(
                        torch.rand(sample_count, device=env.device) < 0.5,
                        -torch.ones(sample_count, device=env.device),
                        torch.ones(sample_count, device=env.device),
                    )
                    mags = torch.empty(sample_count, device=env.device).uniform_(
                        float(min_turn_rad), float(max_turn_rad)
                    )
                    delta_yaw = signs * mags
                new_yaw = base_yaw + delta_yaw
                random_dir_w[:, 0] = torch.cos(new_yaw)
                random_dir_w[:, 1] = torch.sin(new_yaw)
                if torch.any(~prev_valid):
                    fallback_count = int((~prev_valid).sum().item())
                    fallback_yaw = torch.rand(fallback_count, device=env.device) * 2.0 * torch.pi
                    random_dir_w[~prev_valid, 0] = torch.cos(fallback_yaw)
                    random_dir_w[~prev_valid, 1] = torch.sin(fallback_yaw)
                desired_dir_w[resample_mask] = random_dir_w
                env._target_heading_hold_steps[resample_env_ids] = sample_heading_hold_steps(env, sample_count)
                env._target_speed_mps[resample_env_ids] = sample_target_speeds(env, sample_count)
        else:
            target_pos_w = env._desired_pos_w[env_ids]
            drone_pos_w = env._robot.data.root_pos_w[env_ids]
            desired_dir_w = target_pos_w - drone_pos_w
            desired_dir_w[:, 2] = desired_dir_w[:, 2] * env.cfg.moving_target_vertical_dir_scale
            desired_dir_w = desired_dir_w / torch.clamp(torch.linalg.norm(desired_dir_w, dim=1, keepdim=True), min=1e-6)
            safe_prev_dir_w[first_update] = desired_dir_w[first_update]

        if getattr(env.cfg, "moving_target_no_instant_reverse", True):
            dot_prev_desired = torch.sum(safe_prev_dir_w * desired_dir_w, dim=1, keepdim=True)
            reverse_mask = dot_prev_desired.squeeze(-1) < 0.0
            if torch.any(reverse_mask):
                adjusted_dir = desired_dir_w[reverse_mask] - dot_prev_desired[reverse_mask] * safe_prev_dir_w[reverse_mask]
                adjusted_norm = torch.linalg.norm(adjusted_dir, dim=1, keepdim=True)
                adjusted_dir = adjusted_dir / torch.clamp(adjusted_norm, min=1e-6)
                degenerate = adjusted_norm.squeeze(-1) < 1e-6
                if torch.any(degenerate):
                    adjusted_dir[degenerate] = safe_prev_dir_w[reverse_mask][degenerate]
                desired_dir_w[reverse_mask] = adjusted_dir

        turn_rate_limit = float(getattr(env.cfg, "moving_target_turn_rate_limit", 0.0))
        if turn_rate_limit > 0.0:
            max_turn_angle = turn_rate_limit * env.step_dt
            dot_prev_desired = torch.sum(safe_prev_dir_w * desired_dir_w, dim=1).clamp(-1.0, 1.0)
            turn_angle = torch.acos(dot_prev_desired)
            blend = torch.ones_like(turn_angle)
            over_limit = turn_angle > max_turn_angle
            blend[over_limit] = max_turn_angle / torch.clamp(turn_angle[over_limit], min=1e-6)
            new_dir_w = (1.0 - blend.unsqueeze(-1)) * safe_prev_dir_w + blend.unsqueeze(-1) * desired_dir_w
            new_dir_w = new_dir_w / torch.clamp(torch.linalg.norm(new_dir_w, dim=1, keepdim=True), min=1e-6)
            new_dir_w[first_update] = desired_dir_w[first_update]
        else:
            new_dir_w = desired_dir_w

    new_dir_w = apply_scene_obstacle_avoidance(env, env_ids, new_dir_w, motion_mode)
    env._target_dir_w[env_ids] = new_dir_w
    env._target_velocity_w[env_ids] = new_dir_w * env._target_speed_mps[env_ids].unsqueeze(-1)
    if env.cfg.moving_target_z_wave_amplitude > 0.0:
        phase_rate = 2.0 * torch.pi / env.cfg.moving_target_z_wave_period_s
        env._target_z_phase[env_ids] = env._target_z_phase[env_ids] + phase_rate * env.step_dt
        env._target_velocity_w[env_ids, 2] += torch.sin(env._target_z_phase[env_ids]) * env.cfg.moving_target_z_wave_amplitude
    env._desired_pos_w[env_ids] = env._desired_pos_w[env_ids] + env._target_velocity_w[env_ids] * env.step_dt

    height_offset = env._get_visual_altitude_offset()
    moving_target_z_min = float(env.cfg.moving_target_z_min) + height_offset
    moving_target_z_max = float(env.cfg.moving_target_z_max) + height_offset
    below = env._desired_pos_w[env_ids, 2] < moving_target_z_min
    above = env._desired_pos_w[env_ids, 2] > moving_target_z_max
    if torch.any(below):
        env._desired_pos_w[env_ids[below], 2] = moving_target_z_min
        env._target_velocity_w[env_ids[below], 2] = torch.abs(env._target_velocity_w[env_ids[below], 2])
    if torch.any(above):
        env._desired_pos_w[env_ids[above], 2] = moving_target_z_max
        env._target_velocity_w[env_ids[above], 2] = -torch.abs(env._target_velocity_w[env_ids[above], 2])


def moving_pre_physics_step(env) -> None:
    update_moving_targets(env, env._robot._ALL_INDICES)
    speed_mps = torch.linalg.norm(env._robot.data.root_lin_vel_w, dim=1)
    env._episode_speed_sum += speed_mps
    env._episode_speed_steps += 1.0
    env._episode_speed_max = torch.maximum(env._episode_speed_max, speed_mps)
    root_z = env._get_logical_root_z()
    env._episode_min_height = torch.minimum(env._episode_min_height, root_z)
    distance_to_goal = torch.linalg.norm(env._desired_pos_w - env._robot.data.root_pos_w, dim=1)
    env._episode_min_distance = torch.minimum(env._episode_min_distance, distance_to_goal)
