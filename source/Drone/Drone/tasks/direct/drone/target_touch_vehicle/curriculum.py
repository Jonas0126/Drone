from __future__ import annotations


def vehicle_init(env) -> None:
    """Initialize vehicle-specific curriculum counters."""
    env._vector_step_count = 0
    env._vehicle_target_dist_stage_idx = -1


def vehicle_pre_physics_step(env) -> None:
    """Advance the vehicle distance curriculum clock before delegating to the base env."""
    env._vector_step_count += 1


def resolve_vehicle_target_dist_range(env):
    """Resolve the current vehicle target-distance curriculum stage from vector steps."""
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

    if stage_idx != env._vehicle_target_dist_stage_idx:
        env._vehicle_target_dist_stage_idx = stage_idx
        print(
            f"[CURRICULUM][VehicleTargetDist][vector_steps] step={env._vector_step_count} "
            f"stage={stage_idx + 1}/{stage_count} range={min_dist:.1f}~{max_dist:.1f}m",
            flush=True,
        )

    return min_dist, max_dist


def vehicle_reset_idx_impl(env, env_ids, spread_episode_resets: bool, base_reset_impl) -> None:
    """Run the vehicle-specific distance curriculum, then delegate to the target-touch reset flow."""
    original_min = getattr(env.cfg, "target_spawn_distance_min", None)
    original_max = getattr(env.cfg, "target_spawn_distance_max", None)
    original_enabled = bool(getattr(env.cfg, "target_distance_curriculum_enabled", False))

    try:
        stage_range = resolve_vehicle_target_dist_range(env)
        if stage_range is not None:
            env.cfg.target_spawn_distance_min = float(stage_range[0])
            env.cfg.target_spawn_distance_max = float(stage_range[1])
            env.cfg.target_distance_curriculum_enabled = False

        base_reset_impl(env_ids, spread_episode_resets)
    finally:
        env.cfg.target_spawn_distance_min = original_min
        env.cfg.target_spawn_distance_max = original_max
        env.cfg.target_distance_curriculum_enabled = original_enabled
