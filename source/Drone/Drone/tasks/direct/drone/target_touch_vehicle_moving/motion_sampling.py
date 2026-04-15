from __future__ import annotations

import torch


def sample_target_speeds(env, count: int) -> torch.Tensor:
    """Sample per-env moving-target base speeds without changing cfg semantics."""
    fixed_speed_cfg = getattr(env.cfg, "test_fixed_target_speed", None)
    if fixed_speed_cfg is not None:
        return torch.full((count,), float(fixed_speed_cfg), device=env.device)
    speed_min_cfg = getattr(env.cfg, "moving_target_speed_min", None)
    speed_max_cfg = getattr(env.cfg, "moving_target_speed_max", None)
    if speed_min_cfg is None or speed_max_cfg is None:
        return torch.full((count,), float(env.cfg.moving_target_speed), device=env.device)
    speed_min = float(speed_min_cfg)
    speed_max = float(speed_max_cfg)
    if speed_min > speed_max:
        speed_min, speed_max = speed_max, speed_min
    if abs(speed_max - speed_min) < 1e-6:
        return torch.full((count,), speed_min, device=env.device)
    return torch.empty(count, device=env.device).uniform_(speed_min, speed_max)


def sample_heading_hold_steps(env, count: int) -> torch.Tensor:
    """Sample heading-hold duration in env steps."""
    hold_min_s = max(float(getattr(env.cfg, "moving_target_heading_hold_min_s", 2.0)), env.step_dt)
    hold_max_s = max(float(getattr(env.cfg, "moving_target_heading_hold_max_s", hold_min_s)), hold_min_s)
    steps_min = max(1, int(round(hold_min_s / env.step_dt)))
    steps_max = max(steps_min, int(round(hold_max_s / env.step_dt)))
    return torch.randint(steps_min, steps_max + 1, (count,), device=env.device)


def sample_turn_decision_steps(env, count: int) -> torch.Tensor:
    """Sample road-like straight/turn decision duration in env steps."""
    hold_min_s = max(float(getattr(env.cfg, "moving_target_turn_decision_min_s", 2.0)), env.step_dt)
    hold_max_s = max(float(getattr(env.cfg, "moving_target_turn_decision_max_s", hold_min_s)), hold_min_s)
    steps_min = max(1, int(round(hold_min_s / env.step_dt)))
    steps_max = max(steps_min, int(round(hold_max_s / env.step_dt)))
    return torch.randint(steps_min, steps_max + 1, (count,), device=env.device)


def sample_turn_segment_steps(env, count: int) -> torch.Tensor:
    """Sample road-like active turning segment duration in env steps."""
    hold_min_s = max(float(getattr(env.cfg, "moving_target_turn_segment_min_s", 2.0)), env.step_dt)
    hold_max_s = max(float(getattr(env.cfg, "moving_target_turn_segment_max_s", hold_min_s)), hold_min_s)
    steps_min = max(1, int(round(hold_min_s / env.step_dt)))
    steps_max = max(steps_min, int(round(hold_max_s / env.step_dt)))
    return torch.randint(steps_min, steps_max + 1, (count,), device=env.device)


def rotate_dirs_xy(dirs_w: torch.Tensor, delta_yaw_rad: torch.Tensor) -> torch.Tensor:
    """Rotate direction vectors in XY while preserving Z."""
    rotated = dirs_w.clone()
    cos_yaw = torch.cos(delta_yaw_rad)
    sin_yaw = torch.sin(delta_yaw_rad)
    x = dirs_w[:, 0]
    y = dirs_w[:, 1]
    rotated[:, 0] = x * cos_yaw - y * sin_yaw
    rotated[:, 1] = x * sin_yaw + y * cos_yaw
    xy_norm = torch.linalg.norm(rotated[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
    rotated[:, :2] = rotated[:, :2] / xy_norm
    return rotated
