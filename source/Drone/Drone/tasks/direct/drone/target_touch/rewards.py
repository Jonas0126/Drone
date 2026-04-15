from __future__ import annotations

import torch

from isaaclab.utils.math import subtract_frame_transforms


def compute_rewards(env) -> torch.Tensor:
    """Compute the target-touch reward without altering the reward decomposition."""
    lin_vel = torch.sum(torch.square(env._robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(env._robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(env._desired_pos_w - env._robot.data.root_pos_w, dim=1)
    distance_to_goal_tanh_scale = max(float(getattr(env.cfg, "distance_to_goal_tanh_scale", 0.8)), 1e-6)
    distance_to_goal_mapped = 1.0 - torch.tanh(distance_to_goal / distance_to_goal_tanh_scale)
    desired_pos_b, _ = subtract_frame_transforms(
        env._robot.data.root_pos_w,
        env._robot.data.root_quat_w,
        env._desired_pos_w,
    )
    goal_dir_b = desired_pos_b / (torch.linalg.norm(desired_pos_b, dim=1, keepdim=True) + 1e-6)
    approach_speed = torch.sum(env._robot.data.root_lin_vel_b * goal_dir_b, dim=1)
    approach_reward_scale = getattr(env.cfg, "approach_reward_scale", 0.0)
    approach_reward = approach_reward_scale * torch.clamp(approach_speed, min=0.0) * env.step_dt

    if bool(getattr(env.cfg, "progress_reward_best_so_far_only", False)):
        progress_to_goal = torch.clamp(env._best_distance_to_goal - distance_to_goal, min=0.0)
    else:
        progress_to_goal = env._prev_distance_to_goal - distance_to_goal
    if bool(getattr(env.cfg, "progress_reward_normalize_by_initial_distance", False)):
        progress_init_dist_min = max(float(getattr(env.cfg, "progress_reward_initial_distance_min", 10.0)), 1e-6)
        progress_den = torch.clamp(env._last_respawn_target_dist, min=progress_init_dist_min)
        progress_to_goal = progress_to_goal / progress_den
    progress_reward_scale = float(getattr(env.cfg, "progress_reward_scale", 0.0))
    progress_reward = progress_reward_scale * progress_to_goal

    speed_to_goal_reward_scale = float(getattr(env.cfg, "speed_to_goal_reward_scale", 0.0))
    speed_to_goal_reward = (
        speed_to_goal_reward_scale
        * torch.clamp(approach_speed, min=0.0)
        * distance_to_goal_mapped
        * env.step_dt
    )

    tcmd_lambda_4 = float(getattr(env.cfg, "tcmd_lambda_4", 0.0))
    tcmd_lambda_5 = float(getattr(env.cfg, "tcmd_lambda_5", 0.0))
    a_omega_t = torch.linalg.norm(env._actions[:, 1:], dim=1)
    a_delta_sq = torch.sum(torch.square(env._actions - env._prev_actions), dim=1)
    tcmd = (tcmd_lambda_4 * a_omega_t + tcmd_lambda_5 * a_delta_sq) * env.step_dt
    tcmd_penalty = -tcmd

    near_touch_scale = torch.clamp(
        distance_to_goal / env.cfg.near_touch_outer_radius,
        min=env.cfg.near_touch_vel_penalty_min_scale,
        max=1.0,
    )

    touch_threshold = env._get_touch_threshold()
    touched = distance_to_goal <= touch_threshold
    near_touch_distance_reward_ratio = float(getattr(env.cfg, "near_touch_distance_reward_ratio", 1.0))
    near_touch_distance_reward_ratio = min(max(near_touch_distance_reward_ratio, 0.0), 1.0)
    near_touch_not_touched = (distance_to_goal < env.cfg.near_touch_outer_radius) & (~touched)
    distance_to_goal_scale = torch.ones_like(distance_to_goal)
    distance_to_goal_scale[near_touch_not_touched] = near_touch_distance_reward_ratio
    distance_to_goal_reward = (
        distance_to_goal_mapped
        * distance_to_goal_scale
        * env.cfg.distance_to_goal_reward_scale
        * env.step_dt
    )

    touch_bonus = (
        env.cfg.touch_bonus_reward * touched.float()
        if env.cfg.enable_touch_reward
        else torch.zeros_like(distance_to_goal)
    )
    remaining_frac = 1.0 - (env.episode_length_buf.float() / float(env.max_episode_length))
    touch_early_bonus_scale = getattr(env.cfg, "touch_early_bonus_scale", 0.0)
    touch_early_bonus = (
        touch_early_bonus_scale * remaining_frac * touched.float()
        if env.cfg.enable_touch_reward
        else torch.zeros_like(distance_to_goal)
    )

    time_penalty = -env.cfg.time_penalty_scale * env.step_dt * torch.ones_like(distance_to_goal)
    distance_penalty = -env.cfg.distance_penalty_scale * distance_to_goal * env.step_dt
    if bool(getattr(env.cfg, "distance_penalty_only_when_not_approaching", False)):
        not_approaching = (approach_speed <= 0.0).float()
        distance_penalty = distance_penalty * not_approaching

    lin_speed = torch.linalg.norm(env._robot.data.root_lin_vel_b, dim=1)
    hovering_near_touch = (
        (distance_to_goal > touch_threshold)
        & (distance_to_goal < env.cfg.near_touch_outer_radius)
        & (lin_speed < env.cfg.near_touch_hover_speed_threshold)
    )
    near_touch_hover_penalty_scale = getattr(env.cfg, "near_touch_hover_penalty", 0.0)
    near_touch_hover_penalty = -near_touch_hover_penalty_scale * hovering_near_touch.float() * env.step_dt

    near_touch_push_reward_scale = float(getattr(env.cfg, "near_touch_push_reward_scale", 0.0))
    near_touch_zone = (
        (distance_to_goal > touch_threshold)
        & (distance_to_goal < env.cfg.near_touch_outer_radius)
        & (approach_speed > 0.0)
    )
    near_touch_zone_span = max(float(env.cfg.near_touch_outer_radius - touch_threshold), 1e-6)
    near_touch_closeness = torch.clamp(
        1.0 - (distance_to_goal - touch_threshold) / near_touch_zone_span,
        min=0.0,
        max=1.0,
    )
    near_touch_push_reward = (
        near_touch_push_reward_scale * near_touch_closeness * near_touch_zone.float() * env.step_dt
    )

    follow_behind_penalty_scale = float(getattr(env.cfg, "follow_behind_penalty_scale", 0.0))
    follow_behind_outer_radius = float(
        getattr(env.cfg, "follow_behind_outer_radius", env.cfg.near_touch_outer_radius)
    )
    follow_behind_outer_radius = max(follow_behind_outer_radius, touch_threshold + 1e-6)
    follow_behind_min_approach_speed = float(getattr(env.cfg, "follow_behind_min_approach_speed", 0.0))
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
        * env.step_dt
    )

    died_by_height = env._get_logical_root_z() < float(getattr(env.cfg, "died_height_threshold", 0.3))
    died_by_ground = env._get_ground_contact_mask()
    died_by_tilt = env._get_tilt_exceeded_mask()
    died = died_by_height | died_by_ground | died_by_tilt
    far_away = distance_to_goal > env.cfg.far_away_termination_distance
    time_out = env.episode_length_buf >= env.max_episode_length - 1
    failed_no_touch = time_out & (~touched) & (~died) & (~far_away)
    non_tilt_died = died_by_height | died_by_ground
    death_penalty = -env.cfg.death_penalty * non_tilt_died.float()
    tilt_death_penalty_scale = float(getattr(env.cfg, "tilt_death_penalty", env.cfg.death_penalty))
    tilt_death_penalty = -tilt_death_penalty_scale * died_by_tilt.float()
    timeout_penalty = -env.cfg.death_penalty * time_out.float()
    far_away_penalty_scale = float(getattr(env.cfg, "far_away_penalty", env.cfg.death_penalty))
    far_away_penalty = -far_away_penalty_scale * far_away.float()
    failure_penalty_scale = float(getattr(env.cfg, "failure_penalty", env.cfg.death_penalty))
    failure_penalty = -failure_penalty_scale * failed_no_touch.float()

    tilt_forward_reward_scale = float(getattr(env.cfg, "tilt_forward_reward_scale", 0.0))
    gravity_b = env._robot.data.projected_gravity_b
    g_norm = torch.linalg.norm(gravity_b, dim=1).clamp_min(1e-6)
    cos_tilt = torch.clamp((-gravity_b[:, 2]) / g_norm, -1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
    tilt_min_deg = float(getattr(env.cfg, "tilt_target_min_deg", 30.0))
    tilt_max_deg = float(getattr(env.cfg, "tilt_target_max_deg", 35.0))
    if tilt_min_deg > tilt_max_deg:
        tilt_min_deg, tilt_max_deg = tilt_max_deg, tilt_min_deg
    tilt_sigma_deg = max(float(getattr(env.cfg, "tilt_outside_sigma_deg", 10.0)), 1e-3)
    tilt_below_ratio = float(getattr(env.cfg, "tilt_below_reward_ratio", 0.0))
    tilt_below_ratio = min(max(tilt_below_ratio, 0.0), 1.0)
    in_target_band = (tilt_deg >= tilt_min_deg) & (tilt_deg <= tilt_max_deg)
    below_tilt_deg = torch.clamp(tilt_min_deg - tilt_deg, min=0.0)
    below_tilt_reward = tilt_below_ratio * torch.exp(-torch.square(below_tilt_deg / tilt_sigma_deg))
    over_tilt_deg = torch.clamp(tilt_deg - tilt_max_deg, min=0.0)
    over_tilt_penalty = 1.0 - torch.exp(-torch.square(over_tilt_deg / tilt_sigma_deg))
    moving_toward_goal = (approach_speed > 0.0).float()
    tilt_forward_reward = (
        tilt_forward_reward_scale
        * (in_target_band.float() + below_tilt_reward - over_tilt_penalty)
        * moving_toward_goal
        * env.step_dt
    )

    tilt_excess_penalty_scale = float(getattr(env.cfg, "tilt_excess_penalty_scale", 0.0))
    tilt_limit_deg = float(getattr(env.cfg, "max_tilt_deg", 35.0))
    tilt_excess_ratio = torch.clamp((tilt_deg - tilt_limit_deg) / max(tilt_limit_deg, 1e-6), min=0.0)
    tilt_excess_penalty = -tilt_excess_penalty_scale * tilt_excess_ratio * env.step_dt

    rewards = {
        "lin_vel": lin_vel * near_touch_scale * env.cfg.lin_vel_reward_scale * env.step_dt,
        "ang_vel": ang_vel * near_touch_scale * env.cfg.ang_vel_reward_scale * env.step_dt,
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
    reward = torch.sum(
        torch.stack([value for key, value in rewards.items() if key != "timeout_penalty"]),
        dim=0,
    )

    env._prev_distance_to_goal = distance_to_goal.detach()
    env._best_distance_to_goal = torch.minimum(env._best_distance_to_goal, distance_to_goal.detach())
    env._prev_actions.copy_(env._actions.detach())
    for key, value in rewards.items():
        env._episode_sums[key] += value
    return reward
