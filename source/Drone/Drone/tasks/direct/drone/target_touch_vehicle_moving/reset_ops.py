from __future__ import annotations

import torch

from .moving_target import (
    push_targets_out_of_scene_obstacles,
    sample_heading_hold_steps,
    sample_target_speeds,
    sample_turn_decision_steps,
)


def resolve_moving_target_dist_curriculum_range(env) -> tuple[float, float] | None:
    """Resolve the active moving-target distance curriculum stage, if enabled."""
    if not bool(getattr(env.cfg, "target_distance_curriculum_enabled", False)):
        return None

    mode = str(getattr(env.cfg, "target_distance_curriculum_mode", "")).lower()
    if mode not in ("vector_step", "vector_steps", "timestep", "timesteps"):
        return None

    stages_cfg = getattr(env.cfg, "target_distance_curriculum_stages", None)
    if not stages_cfg:
        return None

    stages = [(float(s[0]), float(s[1])) for s in stages_cfg]
    stage_count = len(stages)
    stage_idx = stage_count - 1

    end_steps_cfg = getattr(env.cfg, "target_distance_curriculum_stage_end_steps", None)
    if end_steps_cfg and len(end_steps_cfg) == stage_count - 1:
        end_steps = [max(0, int(x)) for x in end_steps_cfg]
        for i, end_step in enumerate(end_steps):
            if env._vector_step_count < end_step:
                stage_idx = i
                break

    min_dist, max_dist = stages[stage_idx]
    if min_dist > max_dist:
        min_dist, max_dist = max_dist, min_dist

    if stage_idx != env._moving_target_dist_curriculum_stage_idx:
        env._moving_target_dist_curriculum_stage_idx = stage_idx
        print(
            f"[CURRICULUM][VehicleMovingTargetDist][vector_steps] step={env._vector_step_count} "
            f"stage={stage_idx + 1}/{stage_count} range={min_dist:.1f}~{max_dist:.1f}m",
            flush=True,
        )
    return min_dist, max_dist


def moving_reset_idx_impl(env, env_ids: torch.Tensor | None, spread_episode_resets: bool, base_reset_impl) -> None:
    """Reset moving-target state while preserving the existing reset semantics."""
    original_min = getattr(env.cfg, "target_spawn_distance_min", None)
    original_max = getattr(env.cfg, "target_spawn_distance_max", None)
    original_enabled = bool(getattr(env.cfg, "target_distance_curriculum_enabled", False))
    try:
        stage_range = resolve_moving_target_dist_curriculum_range(env)
        if stage_range is not None:
            env.cfg.target_spawn_distance_min = float(stage_range[0])
            env.cfg.target_spawn_distance_max = float(stage_range[1])
            env.cfg.target_distance_curriculum_enabled = False
        base_reset_impl(env_ids, spread_episode_resets)
    finally:
        env.cfg.target_spawn_distance_min = original_min
        env.cfg.target_spawn_distance_max = original_max
        env.cfg.target_distance_curriculum_enabled = original_enabled

    if env_ids is None or len(env_ids) == env.num_envs:
        env_ids = env._robot._ALL_INDICES

    env._target_velocity_w[env_ids] = 0.0
    env._target_dir_w[env_ids] = 0.0
    env._target_base_speed_mps[env_ids] = sample_target_speeds(env, len(env_ids))
    env._target_speed_mps[env_ids] = env._target_base_speed_mps[env_ids]
    env._target_heading_hold_steps[env_ids] = sample_heading_hold_steps(env, len(env_ids))
    env._target_yaw_w[env_ids] = torch.rand(len(env_ids), device=env.device) * 2.0 * torch.pi
    env._target_turn_rate_rad_s[env_ids] = 0.0
    env._target_turn_decision_steps[env_ids] = sample_turn_decision_steps(env, len(env_ids))
    env._target_z_phase[env_ids] = torch.rand(len(env_ids), device=env.device) * 2.0 * torch.pi
    if bool(getattr(env.cfg, "scene_obstacle_spawn_clearance_enabled", False)):
        push_targets_out_of_scene_obstacles(env, env_ids)
    env._episode_start_distance[env_ids] = torch.linalg.norm(
        env._desired_pos_w[env_ids] - env._robot.data.root_pos_w[env_ids], dim=1
    )
    env._episode_min_distance[env_ids] = float("inf")
    env._episode_min_height[env_ids] = float("inf")
    env._episode_speed_sum[env_ids] = 0.0
    env._episode_speed_steps[env_ids] = 0.0
    env._episode_speed_max[env_ids] = 0.0
