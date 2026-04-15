from __future__ import annotations

import torch

from isaaclab.utils.math import subtract_frame_transforms

from .scene_ops import enforce_scene_anchor_clearance, sample_scene_anchor_spawn_xy


def _resolve_reset_env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None or len(env_ids) == env.num_envs:
        return env._robot._ALL_INDICES
    return env_ids


def _collect_reset_metrics(env, env_ids: torch.Tensor) -> dict[str, object]:
    final_distance_to_goal = torch.linalg.norm(
        env._desired_pos_w[env_ids] - env._robot.data.root_pos_w[env_ids], dim=1
    ).mean()
    distance_to_goal_all = torch.linalg.norm(
        env._desired_pos_w[env_ids] - env._robot.data.root_pos_w[env_ids], dim=1
    )
    touched_mask = distance_to_goal_all <= env._get_touch_threshold()
    far_away_mask = distance_to_goal_all > env.cfg.far_away_termination_distance
    died_by_height_mask = env._get_logical_root_z(env_ids) < float(getattr(env.cfg, "died_height_threshold", 0.3))
    died_by_tilt_mask = env._get_tilt_exceeded_mask(env_ids)
    died_mask = died_by_height_mask | env._get_ground_contact_mask(env_ids) | died_by_tilt_mask
    desired_pos_b_env, _ = subtract_frame_transforms(
        env._robot.data.root_pos_w[env_ids],
        env._robot.data.root_quat_w[env_ids],
        env._desired_pos_w[env_ids],
    )
    goal_dir_b_env = desired_pos_b_env / (torch.linalg.norm(desired_pos_b_env, dim=1, keepdim=True) + 1e-6)
    approach_speed_env = torch.sum(env._robot.data.root_lin_vel_b[env_ids] * goal_dir_b_env, dim=1)
    approaching_mask = approach_speed_env > 0.0
    return {
        "final_distance_to_goal": final_distance_to_goal,
        "touched_mask": touched_mask,
        "far_away_mask": far_away_mask,
        "died_mask": died_mask,
        "approaching_mask": approaching_mask,
    }


def _write_episode_logs(env, env_ids: torch.Tensor, metrics: dict[str, object]) -> None:
    extras = {}
    for key in env._episode_sums.keys():
        episodic_sum_avg = torch.mean(env._episode_sums[key][env_ids])
        extras["Episode_RewardRaw/" + key] = episodic_sum_avg
        env._episode_sums[key][env_ids] = 0.0

    env.extras["log"] = {}
    env.extras["log"].update(extras)
    env.extras["log"].update(
        {
            "Episode_Termination/died": torch.count_nonzero(env.reset_terminated[env_ids]).item(),
            "Episode_Termination/time_out": torch.count_nonzero(env.reset_time_outs[env_ids]).item(),
            "Episode_Termination/touched": torch.count_nonzero(env._term_touched[env_ids]).item(),
            "Episode_Termination/failed_no_touch": torch.count_nonzero(
                env.reset_time_outs[env_ids] & (~env._term_touched[env_ids])
            ).item(),
            "Metrics/final_distance_to_goal": metrics["final_distance_to_goal"].item(),
            "Metrics/approaching_rate": torch.mean(metrics["approaching_mask"].float()).item(),
            "Metrics/touched_rate": torch.mean(metrics["touched_mask"].float()).item(),
            "Metrics/far_away_rate": torch.mean(metrics["far_away_mask"].float()).item(),
            "Metrics/died_rate": torch.mean(metrics["died_mask"].float()).item(),
        }
    )


def _prepare_target_spawn_bounds(env) -> tuple[float, float]:
    target_spawn_z_min = float(getattr(env.cfg, "target_spawn_z_min", 0.5))
    target_spawn_z_max = float(getattr(env.cfg, "target_spawn_z_max", 5.0))
    if target_spawn_z_min > target_spawn_z_max:
        target_spawn_z_min, target_spawn_z_max = target_spawn_z_max, target_spawn_z_min
    return target_spawn_z_min, target_spawn_z_max


def _reset_episode_state(env, env_ids: torch.Tensor, spread_episode_resets: bool, base_reset_fn) -> None:
    env._robot.reset(env_ids)
    base_reset_fn(env_ids)
    if spread_episode_resets and len(env_ids) == env.num_envs:
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    env._actions[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0


def _sample_default_target_positions(env, env_ids: torch.Tensor, target_spawn_z_min: float, target_spawn_z_max: float) -> None:
    env._desired_pos_w[env_ids, :2] = torch.zeros_like(env._desired_pos_w[env_ids, :2]).uniform_(-5.0, 5.0)
    env._desired_pos_w[env_ids, :2] += env._terrain.env_origins[env_ids, :2]
    env._desired_pos_w[env_ids, 2] = torch.zeros_like(env._desired_pos_w[env_ids, 2]).uniform_(
        target_spawn_z_min, target_spawn_z_max
    )


def _resolve_spawn_curriculum(env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    joint_pos = env._robot.data.default_joint_pos[env_ids]
    joint_vel = env._robot.data.default_joint_vel[env_ids]
    default_root_state = env._robot.data.default_root_state[env_ids]
    default_root_state[:, :3] += env._terrain.env_origins[env_ids]
    spawn_z_max = env.cfg.spawn_z_max
    spawn_xy_min = env.cfg.spawn_xy_min
    spawn_xy_max = env.cfg.spawn_xy_max
    target_dist_curriculum_enabled = bool(getattr(env.cfg, "target_distance_curriculum_enabled", False))
    if getattr(env.cfg, "curriculum_enabled", False):
        env._curriculum_step += int(len(env_ids))
    if target_dist_curriculum_enabled:
        env._target_dist_curriculum_step += 1
    if getattr(env.cfg, "curriculum_enabled", False):
        ramp = max(1, int(getattr(env.cfg, "curriculum_ramp_steps", 1)))
        frac = min(float(env._curriculum_step) / float(ramp), 1.0)
        if hasattr(env.cfg, "curriculum_spawn_z_max_start") and hasattr(env.cfg, "curriculum_spawn_z_max_end"):
            start = float(getattr(env.cfg, "curriculum_spawn_z_max_start"))
            end = float(getattr(env.cfg, "curriculum_spawn_z_max_end"))
            spawn_z_max = max(float(env.cfg.spawn_z_min), start + (end - start) * frac)
        else:
            stages = list(getattr(env.cfg, "curriculum_spawn_max_stages", (spawn_z_max,)))
            stage_idx = min(int(frac * len(stages)), len(stages) - 1)
            if not hasattr(env, "_curriculum_stage_idx"):
                env._curriculum_stage_idx = -1
            if stage_idx != env._curriculum_stage_idx:
                env._curriculum_stage_idx = stage_idx
                stage_max = float(stages[stage_idx])
                print(
                    f"[CURRICULUM][Touch] stage={stage_idx + 1}/{len(stages)} "
                    f"spawn_range=1~{stage_max}m",
                    flush=True,
                )
            spawn_z_max = max(float(stages[stage_idx]), float(env.cfg.spawn_z_min))
            spawn_xy_min = -float(stages[stage_idx])
            spawn_xy_max = float(stages[stage_idx])
    return joint_pos, joint_vel, default_root_state, float(spawn_z_max), float(spawn_xy_min), float(spawn_xy_max)


def _sample_robot_spawn(env, env_ids: torch.Tensor, default_root_state: torch.Tensor, spawn_z_max: float, spawn_xy_min: float, spawn_xy_max: float) -> None:
    spawn_xy = torch.empty(
        (len(env_ids), 2),
        device=default_root_state.device,
        dtype=default_root_state.dtype,
    ).uniform_(spawn_xy_min, spawn_xy_max)
    spawn_z = torch.empty(
        (len(env_ids),),
        device=default_root_state.device,
        dtype=default_root_state.dtype,
    ).uniform_(env.cfg.spawn_z_min, spawn_z_max)
    height_offset = env._get_visual_altitude_offset()
    if bool(getattr(env.cfg, "scene_anchor_enabled", False)) and env._scene_anchor_centers_w is not None:
        anchor_spawn_xy = sample_scene_anchor_spawn_xy(env, env_ids, dtype=default_root_state.dtype)
        default_root_state[:, 0] = anchor_spawn_xy[:, 0]
        default_root_state[:, 1] = anchor_spawn_xy[:, 1]
    else:
        default_root_state[:, 0] = env._terrain.env_origins[env_ids, 0] + spawn_xy[:, 0]
        default_root_state[:, 1] = env._terrain.env_origins[env_ids, 1] + spawn_xy[:, 1]
    default_root_state[:, 2] = env._terrain.env_origins[env_ids, 2] + spawn_z + height_offset
    env._observation_local_origin_xy[env_ids] = default_root_state[:, :2]


def _apply_target_distance_curriculum(env, min_dist: float, max_dist: float) -> tuple[float, float]:
    stages = getattr(env.cfg, "target_distance_curriculum_stages", None)
    dist_ramp = int(getattr(env.cfg, "target_distance_curriculum_ramp_steps", getattr(env.cfg, "curriculum_ramp_steps", 1)))
    dist_ramp = max(1, dist_ramp)
    raw_dist_frac = min(float(env._target_dist_curriculum_step) / float(dist_ramp), 1.0)
    progress_power = float(getattr(env.cfg, "target_distance_curriculum_progress_power", 1.0))
    progress_power = max(progress_power, 1e-6)
    dist_frac = min(max(raw_dist_frac**progress_power, 0.0), 1.0)
    if stages:
        stage_pairs = [(float(s[0]), float(s[1])) for s in stages]
        stage_count = len(stage_pairs)
        stage_steps_cfg = getattr(env.cfg, "target_distance_curriculum_stage_steps", None)
        stage_idx = None
        if stage_steps_cfg is not None:
            stage_steps = [max(1, int(x)) for x in stage_steps_cfg]
            if len(stage_steps) == stage_count:
                progress_step = int(getattr(env, "common_step_counter", env._target_dist_curriculum_step))
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
        if not hasattr(env, "_target_dist_curriculum_stage_idx"):
            env._target_dist_curriculum_stage_idx = -1
        if stage_idx != env._target_dist_curriculum_stage_idx:
            env._target_dist_curriculum_stage_idx = stage_idx
            print(
                f"[CURRICULUM][TargetDist] stage={stage_idx + 1}/{stage_count} "
                f"range={min_dist:.1f}~{max_dist:.1f}m",
                flush=True,
            )
        return min_dist, max_dist

    min_start = getattr(env.cfg, "target_distance_curriculum_min_start", None)
    min_end = getattr(env.cfg, "target_distance_curriculum_min_end", None)
    max_start = getattr(env.cfg, "target_distance_curriculum_max_start", None)
    max_end = getattr(env.cfg, "target_distance_curriculum_max_end", None)
    if min_start is None:
        min_start = getattr(env.cfg, "target_distance_curriculum_start", min_dist)
    if max_end is None:
        max_end = getattr(env.cfg, "target_distance_curriculum_end", max_dist)
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
    print_stages = max(1, int(getattr(env.cfg, "target_distance_curriculum_print_stages", 10)))
    stage_idx = min(int(dist_frac * print_stages), print_stages - 1)
    if not hasattr(env, "_target_dist_curriculum_stage_idx"):
        env._target_dist_curriculum_stage_idx = -1
    if stage_idx != env._target_dist_curriculum_stage_idx:
        env._target_dist_curriculum_stage_idx = stage_idx
        print(
            f"[CURRICULUM][TargetDist] stage={stage_idx + 1}/{print_stages} "
            f"range={min_dist:.1f}~{max_dist:.1f}m",
            flush=True,
        )
    return min_dist, max_dist


def _sample_target_positions(env, env_ids: torch.Tensor, default_root_state: torch.Tensor, target_spawn_z_min: float, target_spawn_z_max: float) -> None:
    height_offset = env._get_visual_altitude_offset()
    target_dist_min = getattr(env.cfg, "target_spawn_distance_min", None)
    target_dist_max = getattr(env.cfg, "target_spawn_distance_max", None)
    if target_dist_min is not None and target_dist_max is not None:
        min_dist = float(target_dist_min)
        max_dist = float(target_dist_max)
        if bool(getattr(env.cfg, "target_distance_curriculum_enabled", False)):
            min_dist, max_dist = _apply_target_distance_curriculum(env, min_dist, max_dist)
        if min_dist > max_dist:
            min_dist, max_dist = max_dist, min_dist
        target_theta = torch.empty((len(env_ids),), device=default_root_state.device, dtype=default_root_state.dtype).uniform_(
            0.0, 2.0 * torch.pi
        )
        target_radius = torch.empty(
            (len(env_ids),), device=default_root_state.device, dtype=default_root_state.dtype
        ).uniform_(min_dist, max_dist)
        env._desired_pos_w[env_ids, 0] = default_root_state[:, 0] + target_radius * torch.cos(target_theta)
        env._desired_pos_w[env_ids, 1] = default_root_state[:, 1] + target_radius * torch.sin(target_theta)
        env._desired_pos_w[env_ids, 2] = torch.zeros_like(env._desired_pos_w[env_ids, 2]).uniform_(
            target_spawn_z_min, target_spawn_z_max
        )
        env._desired_pos_w[env_ids, 2] += env._terrain.env_origins[env_ids, 2] + height_offset
        if bool(getattr(env.cfg, "scene_anchor_enabled", False)):
            target_extra_clearance = float(getattr(env.cfg, "scene_anchor_target_extra_clearance_m", 0.0))
            env._desired_pos_w[env_ids] = enforce_scene_anchor_clearance(
                env, env._desired_pos_w[env_ids], env_ids, target_extra_clearance
            )
            env._last_respawn_target_dist[env_ids] = torch.linalg.norm(
                env._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
            )
        else:
            env._last_respawn_target_dist[env_ids] = target_radius
        return

    env._desired_pos_w[env_ids, 2] += env._terrain.env_origins[env_ids, 2] + height_offset
    if bool(getattr(env.cfg, "scene_anchor_enabled", False)):
        env._desired_pos_w[env_ids] = enforce_scene_anchor_clearance(
            env, env._desired_pos_w[env_ids], env_ids
        )
    env._last_respawn_target_dist[env_ids] = torch.linalg.norm(
        env._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
    )


def _write_reset_state_to_sim(env, env_ids: torch.Tensor, default_root_state: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
    env._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    env._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    env._prev_distance_to_goal[env_ids] = torch.linalg.norm(
        env._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
    )
    env._best_distance_to_goal[env_ids] = env._prev_distance_to_goal[env_ids]
    env._term_touched[env_ids] = False


def reset_idx_impl(env, env_ids: torch.Tensor | None, spread_episode_resets: bool, base_reset_fn) -> None:
    """Shared target-touch reset flow, split into smaller steps for maintainability."""
    env_ids = _resolve_reset_env_ids(env, env_ids)
    metrics = _collect_reset_metrics(env, env_ids)
    _write_episode_logs(env, env_ids, metrics)
    _reset_episode_state(env, env_ids, spread_episode_resets, base_reset_fn)

    target_spawn_z_min, target_spawn_z_max = _prepare_target_spawn_bounds(env)
    _sample_default_target_positions(env, env_ids, target_spawn_z_min, target_spawn_z_max)

    joint_pos, joint_vel, default_root_state, spawn_z_max, spawn_xy_min, spawn_xy_max = _resolve_spawn_curriculum(env, env_ids)
    _sample_robot_spawn(env, env_ids, default_root_state, spawn_z_max, spawn_xy_min, spawn_xy_max)
    _sample_target_positions(env, env_ids, default_root_state, target_spawn_z_min, target_spawn_z_max)
    _write_reset_state_to_sim(env, env_ids, default_root_state, joint_pos, joint_vel)


def reset_idx_train(env, env_ids: torch.Tensor | None) -> None:
    """Training reset entry-point with staggered episode lengths."""
    reset_idx_impl(env, env_ids, spread_episode_resets=True, base_reset_fn=env._call_base_reset_idx)


def reset_idx_test(env, env_ids: torch.Tensor | None) -> None:
    """Test reset entry-point preserving the existing logging and deterministic reset policy."""
    if not hasattr(env, "_test_reset_counter"):
        env._test_reset_counter = 0
        env._test_total_episodes = 0
        env._test_total_success = 0
        env._test_total_success_steps = 0.0
        print("[TEST][Touch] DroneTargetTouchTestEnv reset hook active", flush=True)
    env._test_reset_counter += 1
    env.cfg.curriculum_enabled = False
    if getattr(env.cfg, "target_distance_curriculum_enabled", False):
        env.cfg.target_distance_curriculum_enabled = False
        env.cfg.target_spawn_distance_min = 20.0
        env.cfg.target_spawn_distance_max = 50.0
    env.cfg.died_height_threshold = 0.1

    log_env_ids = _resolve_reset_env_ids(env, env_ids)
    touched_mask = env._term_touched[log_env_ids]
    touched_count = int(touched_mask.sum().item())
    batch_episodes = int(touched_mask.numel())
    distance_to_goal = torch.linalg.norm(
        env._desired_pos_w[log_env_ids] - env._robot.data.root_pos_w[log_env_ids], dim=1
    )
    died_by_height = env._get_logical_root_z(log_env_ids) < float(getattr(env.cfg, "died_height_threshold", 0.3))
    died_by_ground = env._get_ground_contact_mask(log_env_ids)
    died_by_tilt = env._get_tilt_exceeded_mask(log_env_ids)
    died_mask = died_by_height | died_by_ground | died_by_tilt
    far_mask = distance_to_goal > env.cfg.far_away_termination_distance
    time_out_mask = env.reset_time_outs[log_env_ids]
    fail_no_touch_mask = time_out_mask & (~touched_mask) & (~died_mask) & (~far_mask)
    reason_summary = (
        f"dh:{int(died_by_height.sum().item())},"
        f"dg:{int(died_by_ground.sum().item())},"
        f"dt:{int(died_by_tilt.sum().item())},"
        f"fa:{int(far_mask.sum().item())},"
        f"to:{int(time_out_mask.sum().item())},"
        f"fn:{int(fail_no_touch_mask.sum().item())}"
    )
    env._test_total_episodes += batch_episodes
    env._test_total_success += touched_count
    total_success_rate = (env._test_total_success / env._test_total_episodes) if env._test_total_episodes > 0 else 0.0
    batch_total_reward = torch.zeros(len(log_env_ids), dtype=torch.float, device=env.device)
    for key in env._episode_sums.keys():
        batch_total_reward += env._episode_sums[key][log_env_ids]
    reward_summary = (
        f"reward={float(batch_total_reward.mean().item()):.2f}"
        f"(min:{float(batch_total_reward.min().item()):.2f},"
        f"max:{float(batch_total_reward.max().item()):.2f})"
    )

    reset_idx_impl(env, env_ids, spread_episode_resets=False, base_reset_fn=env._call_base_reset_idx)

    respawn_dist = env._last_respawn_target_dist[log_env_ids]
    respawn_dist_summary = (
        f"rd={float(respawn_dist.mean().item()):.1f}"
        f"(min:{float(respawn_dist.min().item()):.1f},"
        f"max:{float(respawn_dist.max().item()):.1f})"
    )
    if touched_count > 0:
        touched_steps = env.episode_length_buf[log_env_ids][touched_mask].float()
        env._test_total_success_steps += float(touched_steps.sum().item())
        total_avg_steps = env._test_total_success_steps / max(env._test_total_success, 1)
        print(
            "[TEST][Touch] count="
            f"{touched_count}, "
            f"max_steps={float(touched_steps.max().item()):.1f}, "
            f"total_rate={total_success_rate:.3f}, "
            f"total_avg_steps={total_avg_steps:.1f}, "
            f"{reward_summary}, "
            f"{respawn_dist_summary}, "
            f"rsn={reason_summary}",
            flush=True,
        )
        return

    print(
        "[TEST][Touch] count=0, "
        f"total_rate={total_success_rate:.3f}, "
        f"total_avg_steps={env._test_total_success_steps / max(env._test_total_success, 1):.1f}, "
        f"{reward_summary}, "
        f"{respawn_dist_summary}, "
        f"rsn={reason_summary}",
        flush=True,
    )
