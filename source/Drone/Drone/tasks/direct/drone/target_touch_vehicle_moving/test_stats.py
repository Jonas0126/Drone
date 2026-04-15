from __future__ import annotations

import torch

from .test_visuals import append_trail_sample, clear_trails


def compute_test_dones(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute test-env dones while preserving touch-delay semantics."""
    time_out = env.episode_length_buf >= env.max_episode_length - 1

    died_by_height = env._get_logical_root_z() < float(getattr(env.cfg, "died_height_threshold", 0.3))
    died_by_tilt = env._get_tilt_exceeded_mask()
    died = died_by_height | env._get_ground_contact_mask() | died_by_tilt
    distance_to_goal = torch.linalg.norm(env._desired_pos_w - env._robot.data.root_pos_w, dim=1)
    far_away = distance_to_goal > env.cfg.far_away_termination_distance
    touched = distance_to_goal <= env._get_touch_threshold()

    if env._touch_reset_delay_steps > 0 and env.cfg.terminate_on_touch:
        newly_touched = touched & (env._touch_reset_countdown == 0)
        env._touch_reset_countdown[newly_touched] = env._touch_reset_delay_steps
        touch_terminated = env._touch_reset_countdown == 1
        active_countdown = env._touch_reset_countdown > 0
        env._touch_reset_countdown[active_countdown] -= 1
        env._term_touched = touched | active_countdown | touch_terminated
    else:
        touch_terminated = touched if env.cfg.terminate_on_touch else torch.zeros_like(touched)
        env._term_touched = touched

    terminated = died | far_away | touch_terminated
    return terminated, time_out


def reset_test_idx(env, env_ids: torch.Tensor | None) -> None:
    """Run the original moving-test reset hook and keep printed statistics unchanged."""
    _init_test_counters_if_needed(env)
    env._test_reset_counter += 1

    env.cfg.curriculum_enabled = False
    env.cfg.died_height_threshold = 0.1

    if env_ids is None or len(env_ids) == env.num_envs:
        test_env_ids = env._robot._ALL_INDICES
    else:
        test_env_ids = env_ids

    env._touch_reset_countdown[test_env_ids] = 0

    if torch.any(test_env_ids == env._trail_env_id):
        clear_trails(env)

    batch_stats = _compute_batch_stats(env, test_env_ids)
    _accumulate_test_totals(env, batch_stats)
    _print_batch_stats(env, batch_stats)
    if env._test_reset_counter % 100 == 0:
        _print_total_stats(env)

    env._reset_idx_impl(env_ids, spread_episode_resets=True)
    if torch.any(test_env_ids == env._trail_env_id):
        append_trail_sample(env, env._trail_env_id)


def _init_test_counters_if_needed(env) -> None:
    if hasattr(env, "_test_reset_counter"):
        return
    env._test_reset_counter = 0
    env._test_total_episodes = 0
    env._test_total_success = 0
    env._test_total_success_steps = 0.0
    env._test_total_died = 0
    env._test_total_dh = 0
    env._test_total_dg = 0
    env._test_total_dt = 0
    env._test_total_fa = 0
    env._test_total_to = 0
    env._test_total_fn = 0
    print("[TEST][VehicleMovingTouch] reset hook active", flush=True)


def _compute_batch_stats(env, test_env_ids: torch.Tensor) -> dict[str, object]:
    distance_to_goal = torch.linalg.norm(
        env._desired_pos_w[test_env_ids] - env._robot.data.root_pos_w[test_env_ids], dim=1
    )
    died_by_height = env._get_logical_root_z(test_env_ids) < float(getattr(env.cfg, "died_height_threshold", 0.3))
    died_by_contact = env._get_ground_contact_mask(test_env_ids)
    died_by_tilt = env._get_tilt_exceeded_mask(test_env_ids)
    died_mask = died_by_height | died_by_contact | died_by_tilt
    far_mask = distance_to_goal > env.cfg.far_away_termination_distance
    time_out_mask = env.reset_time_outs[test_env_ids]
    touched_mask = env._term_touched[test_env_ids]
    fail_no_touch_mask = time_out_mask & (~touched_mask) & (~died_mask) & (~far_mask)

    touched_count = int(touched_mask.sum().item())
    batch_total_reward = torch.zeros(len(test_env_ids), dtype=torch.float, device=env.device)
    for key in env._episode_sums.keys():
        batch_total_reward += env._episode_sums[key][test_env_ids]

    return {
        "test_env_ids": test_env_ids,
        "touched_mask": touched_mask,
        "touched_count": touched_count,
        "time_out_mask": time_out_mask,
        "died_mask": died_mask,
        "died_by_height": died_by_height,
        "died_by_contact": died_by_contact,
        "died_by_tilt": died_by_tilt,
        "far_mask": far_mask,
        "fail_no_touch_mask": fail_no_touch_mask,
        "batch_total_reward": batch_total_reward,
        "reward_summary": f"reward={float(batch_total_reward.mean().item()):.2f}",
        "start_dist_summary": _format_start_distance_summary(env, test_env_ids),
        "min_dist_summary": _format_min_distance_summary(env, test_env_ids),
        "min_height_summary": _format_min_height_summary(env, test_env_ids),
        "speed_summary": _format_speed_summary(env, test_env_ids),
        "reason_summary": _format_reason_summary(
            died_by_height=died_by_height,
            died_by_contact=died_by_contact,
            died_by_tilt=died_by_tilt,
            far_mask=far_mask,
            time_out_mask=time_out_mask,
            fail_no_touch_mask=fail_no_touch_mask,
        ),
    }


def _format_start_distance_summary(env, test_env_ids: torch.Tensor) -> str:
    start_dist_batch = env._episode_start_distance[test_env_ids]
    finite_start_mask = torch.isfinite(start_dist_batch)
    if not torch.any(finite_start_mask):
        return "start_dist=nan"
    start_dist_values = start_dist_batch[finite_start_mask]
    return f"start_dist={float(start_dist_values.mean().item()):.3f}"


def _format_min_distance_summary(env, test_env_ids: torch.Tensor) -> str:
    min_dist_batch = env._episode_min_distance[test_env_ids]
    return f"closest_dist={float(min_dist_batch.mean().item()):.3f}"


def _format_min_height_summary(env, test_env_ids: torch.Tensor) -> str:
    min_height_batch = env._episode_min_height[test_env_ids]
    finite_min_height_mask = torch.isfinite(min_height_batch)
    if not torch.any(finite_min_height_mask):
        return "lowest_z=nan"
    min_height_values = min_height_batch[finite_min_height_mask]
    return f"lowest_z={float(min_height_values.mean().item()):.3f}"


def _format_speed_summary(env, test_env_ids: torch.Tensor) -> str:
    speed_steps_batch = env._episode_speed_steps[test_env_ids]
    valid_speed_mask = speed_steps_batch > 0.0
    if not torch.any(valid_speed_mask):
        return "speed_avg=nan,speed_max=nan"
    speed_avg_batch = env._episode_speed_sum[test_env_ids][valid_speed_mask] / speed_steps_batch[valid_speed_mask]
    speed_max_batch = env._episode_speed_max[test_env_ids][valid_speed_mask]
    return (
        f"speed_avg={float(speed_avg_batch.mean().item()):.3f}mps"
        f"({float((speed_avg_batch.mean() * 3.6).item()):.2f}kmh),"
        f"speed_max={float(speed_max_batch.mean().item()):.3f}mps"
        f"({float((speed_max_batch.mean() * 3.6).item()):.2f}kmh)"
    )


def _format_reason_summary(
    *,
    died_by_height: torch.Tensor,
    died_by_contact: torch.Tensor,
    died_by_tilt: torch.Tensor,
    far_mask: torch.Tensor,
    time_out_mask: torch.Tensor,
    fail_no_touch_mask: torch.Tensor,
) -> str:
    return (
        f"dh:{int(died_by_height.sum().item())},"
        f"dg:{int(died_by_contact.sum().item())},"
        f"dt:{int(died_by_tilt.sum().item())},"
        f"fa:{int(far_mask.sum().item())},"
        f"to:{int(time_out_mask.sum().item())},"
        f"fn:{int(fail_no_touch_mask.sum().item())}"
    )


def _accumulate_test_totals(env, batch_stats: dict[str, object]) -> None:
    touched_mask = batch_stats["touched_mask"]
    touched_count = int(batch_stats["touched_count"])
    died_mask = batch_stats["died_mask"]
    died_by_height = batch_stats["died_by_height"]
    died_by_contact = batch_stats["died_by_contact"]
    died_by_tilt = batch_stats["died_by_tilt"]
    far_mask = batch_stats["far_mask"]
    time_out_mask = batch_stats["time_out_mask"]
    fail_no_touch_mask = batch_stats["fail_no_touch_mask"]
    test_env_ids = batch_stats["test_env_ids"]

    batch_episodes = int(touched_mask.numel())
    batch_dh = int(died_by_height.sum().item())
    batch_dg = int(died_by_contact.sum().item())
    batch_dt = int(died_by_tilt.sum().item())
    batch_fa = int(far_mask.sum().item())
    batch_to = int(time_out_mask.sum().item())
    batch_fn = int(fail_no_touch_mask.sum().item())
    batch_died = int(died_mask.sum().item())
    env._test_total_episodes += batch_episodes
    env._test_total_success += touched_count
    env._test_total_died += batch_died
    env._test_total_dh += batch_dh
    env._test_total_dg += batch_dg
    env._test_total_dt += batch_dt
    env._test_total_fa += batch_fa
    env._test_total_to += batch_to
    env._test_total_fn += batch_fn

    if touched_count > 0:
        touched_steps = env.episode_length_buf[test_env_ids][touched_mask].float()
        env._test_total_success_steps += float(touched_steps.sum().item())


def _print_batch_stats(env, batch_stats: dict[str, object]) -> None:
    touched_count = int(batch_stats["touched_count"])
    touched_mask = batch_stats["touched_mask"]
    test_env_ids = batch_stats["test_env_ids"]
    total_success_rate = (env._test_total_success / env._test_total_episodes) if env._test_total_episodes > 0 else 0.0
    total_avg_steps = env._test_total_success_steps / max(env._test_total_success, 1)

    if touched_count > 0:
        touched_steps = env.episode_length_buf[test_env_ids][touched_mask].float()
        print(
            "[TEST][VehicleMovingTouch] count="
            f"{touched_count}, "
            f"max_steps={float(touched_steps.max().item()):.1f}, "
            f"total_rate={total_success_rate:.3f}, "
            f"total_avg_steps={total_avg_steps:.1f}, "
            f"{batch_stats['reward_summary']}, "
            f"{batch_stats['start_dist_summary']}, "
            f"{batch_stats['min_dist_summary']}, "
            f"{batch_stats['min_height_summary']}, "
            f"{batch_stats['speed_summary']}, "
            f"rsn={batch_stats['reason_summary']}",
            flush=True,
        )
    else:
        print(
            "[TEST][VehicleMovingTouch] count=0, "
            f"total_rate={total_success_rate:.3f}, "
            f"total_avg_steps={total_avg_steps:.1f}, "
            f"{batch_stats['reward_summary']}, "
            f"{batch_stats['start_dist_summary']}, "
            f"{batch_stats['min_dist_summary']}, "
            f"{batch_stats['min_height_summary']}, "
            f"{batch_stats['speed_summary']}, "
            f"rsn={batch_stats['reason_summary']}",
            flush=True,
        )


def _print_total_stats(env) -> None:
    total_episodes = max(int(env._test_total_episodes), 1)
    total_died = int(env._test_total_died)
    total_died_den = max(total_died, 1)
    total_avg_steps = env._test_total_success_steps / max(env._test_total_success, 1)
    print(
        "[TEST][VehicleMovingTouch][TOTAL@"
        f"{env._test_reset_counter}] "
        f"episodes={env._test_total_episodes}, "
        f"success={env._test_total_success}, "
        f"pass_rate={env._test_total_success / total_episodes:.3f}, "
        f"avg_success_steps={total_avg_steps:.1f}, "
        f"died_total={total_died}, "
        f"died_rate={total_died / total_episodes:.3f}, "
        f"dh={env._test_total_dh}({env._test_total_dh / total_died_den:.3f}), "
        f"dg={env._test_total_dg}({env._test_total_dg / total_died_den:.3f}), "
        f"dt={env._test_total_dt}({env._test_total_dt / total_died_den:.3f}), "
        f"fa={env._test_total_fa}({env._test_total_fa / total_episodes:.3f}), "
        f"to={env._test_total_to}({env._test_total_to / total_episodes:.3f}), "
        f"fn={env._test_total_fn}({env._test_total_fn / total_episodes:.3f})",
        flush=True,
    )
