from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import matrix_from_quat

from ..markers import CUBOID_MARKER_CFG, POSITION_GOAL_MARKER_CFG, SPHERE_MARKER_CFG, VisualizationMarkers


def init_test_mode_state(env) -> None:
    """Initialize test-only buffers and trail sampling state."""
    env._trail_sample_counter = 0
    env._trail_env_id = int(getattr(env.cfg, "test_trail_env_id", 0))
    env._trail_max_points = max(1, int(getattr(env.cfg, "test_trail_max_points", 180)))
    env._trail_sample_steps = max(1, int(getattr(env.cfg, "test_trail_sample_steps", 2)))
    env._target_trail_points = torch.empty((0, 3), dtype=torch.float, device=env.device)
    env._drone_trail_points = torch.empty((0, 3), dtype=torch.float, device=env.device)
    env._touch_reset_delay_steps = max(0, int(getattr(env.cfg, "test_touch_reset_delay_steps", 0)))
    env._touch_reset_countdown = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def set_test_debug_vis(env, debug_vis: bool) -> None:
    """Configure the demo/test visual markers for the moving-target environment."""
    if debug_vis:
        _ensure_goal_visualizer(env)
        _set_optional_outline_vis(env, True)
    else:
        if hasattr(env, "goal_pos_visualizer"):
            env.goal_pos_visualizer.set_visibility(False)
        _set_optional_outline_vis(env, False)

    trail_enabled = bool(getattr(env.cfg, "test_trail_enabled", True))
    if debug_vis and trail_enabled:
        _ensure_trail_visualizer(env)
        env.trail_visualizer.set_visibility(True)
    elif hasattr(env, "trail_visualizer"):
        env.trail_visualizer.set_visibility(False)


def test_debug_vis_callback(env, event) -> None:
    """Update test markers and trail markers each debug-visualization tick."""
    if not hasattr(env, "scene") or not hasattr(env, "_robot"):
        return
    try:
        if hasattr(env, "goal_pos_visualizer"):
            touched = torch.linalg.norm(env._desired_pos_w - env._robot.data.root_pos_w, dim=1) <= env._get_touch_threshold()
            display_touched = touched | (env._touch_reset_countdown > 0)
            marker_indices = display_touched.to(dtype=torch.int32)
            env.goal_pos_visualizer.visualize(env._desired_pos_w, marker_indices=marker_indices)
        if not bool(getattr(env.cfg, "test_trail_enabled", True)) or not hasattr(env, "trail_visualizer"):
            return
        if env._trail_env_id < 0 or env._trail_env_id >= env.num_envs:
            return

        env._trail_sample_counter += 1
        if env._trail_sample_counter % env._trail_sample_steps == 0:
            append_trail_sample(env, env._trail_env_id)

        target_count = int(env._target_trail_points.shape[0])
        drone_count = int(env._drone_trail_points.shape[0])
        total_count = target_count + drone_count
        if total_count == 0:
            env.trail_visualizer.set_visibility(False)
            return

        env.trail_visualizer.set_visibility(True)
        translations = torch.cat([env._target_trail_points, env._drone_trail_points], dim=0)
        marker_indices = torch.cat(
            [
                torch.zeros(target_count, dtype=torch.int32, device=env.device),
                torch.ones(drone_count, dtype=torch.int32, device=env.device),
            ],
            dim=0,
        )
        env.trail_visualizer.visualize(translations=translations, marker_indices=marker_indices)
    except Exception as exc:
        if "invalidated" in str(exc).lower() or "backend" in str(exc).lower():
            return
        raise


def append_trail_sample(env, env_id: int) -> None:
    env_idx = int(env_id)
    target_point = env._desired_pos_w[env_idx : env_idx + 1].detach().clone()
    drone_back_point = compute_drone_back_point(env, env_idx).unsqueeze(0)
    env._target_trail_points = append_point_fifo(env, env._target_trail_points, target_point)
    env._drone_trail_points = append_point_fifo(env, env._drone_trail_points, drone_back_point)


def append_point_fifo(env, history: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    if history.shape[0] >= env._trail_max_points:
        history = history[-(env._trail_max_points - 1) :]
    return torch.cat([history, point], dim=0)


def compute_drone_back_point(env, env_id: int) -> torch.Tensor:
    root_pos_w = env._robot.data.root_pos_w[env_id]
    root_quat_w = env._robot.data.root_quat_w[env_id : env_id + 1]
    rot_mat_wb = matrix_from_quat(root_quat_w).reshape(3, 3)
    forward_w = rot_mat_wb[:, 0]
    back_offset = float(getattr(env.cfg, "test_trail_drone_back_offset", 0.35))
    return root_pos_w - forward_w * back_offset


def clear_trails(env) -> None:
    env._trail_sample_counter = 0
    env._target_trail_points = torch.empty((0, 3), dtype=torch.float, device=env.device)
    env._drone_trail_points = torch.empty((0, 3), dtype=torch.float, device=env.device)
    if hasattr(env, "trail_visualizer"):
        env.trail_visualizer.set_visibility(False)


def _ensure_goal_visualizer(env) -> None:
    if hasattr(env, "goal_pos_visualizer"):
        env.goal_pos_visualizer.set_visibility(True)
        return
    marker_cfg = CUBOID_MARKER_CFG.copy()
    marker_edge_length = getattr(env.cfg, "touch_marker_edge_length", None)
    if marker_edge_length is None:
        touch_diameter = env._get_touch_threshold() * 2.0
        marker_scale = 0.9
        marker_size = touch_diameter * marker_scale
    else:
        marker_size = float(marker_edge_length)
    marker_cfg.markers["cuboid"].size = (marker_size, marker_size, marker_size)
    marker_cfg.markers["cuboid_touched"] = sim_utils.CuboidCfg(
        size=(marker_size, marker_size, marker_size),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
    )
    marker_cfg.prim_path = "/Visuals/Command/goal_position"
    env.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
    env.goal_pos_visualizer.set_visibility(True)


def _set_optional_outline_vis(env, debug_vis: bool) -> None:
    if not bool(getattr(env.cfg, "drone_outline_enabled", False)):
        if hasattr(env, "drone_outline_visualizer"):
            env.drone_outline_visualizer.set_visibility(False)
        return
    if debug_vis:
        if not hasattr(env, "drone_outline_visualizer"):
            marker_cfg = SPHERE_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/drone_outline"
            marker_cfg.markers["sphere"].radius = float(getattr(env.cfg, "drone_outline_radius", 0.28))
            marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(getattr(env.cfg, "drone_outline_color", (0.15, 0.95, 1.0))),
                emissive_color=tuple(getattr(env.cfg, "drone_outline_emissive_color", (0.08, 0.38, 0.42))),
                opacity=float(getattr(env.cfg, "drone_outline_opacity", 0.18)),
                roughness=0.2,
            )
            env.drone_outline_visualizer = VisualizationMarkers(marker_cfg)
        env.drone_outline_visualizer.set_visibility(True)
    elif hasattr(env, "drone_outline_visualizer"):
        env.drone_outline_visualizer.set_visibility(False)


def _ensure_trail_visualizer(env) -> None:
    if hasattr(env, "trail_visualizer"):
        return
    marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
    marker_cfg.prim_path = "/Visuals/Command/vehicle_moving_trails"
    trail_marker_radius = float(getattr(env.cfg, "test_trail_marker_radius", 0.035))
    marker_cfg.markers["target_far"].radius = trail_marker_radius
    marker_cfg.markers["target_near"].radius = trail_marker_radius
    marker_cfg.markers["target_invisible"].radius = trail_marker_radius
    env.trail_visualizer = VisualizationMarkers(marker_cfg)
