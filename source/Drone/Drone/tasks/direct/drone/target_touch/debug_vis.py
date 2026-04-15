from __future__ import annotations

from ..markers import CUBOID_MARKER_CFG, VisualizationMarkers


def set_debug_vis(env, debug_vis: bool) -> None:
    """Create or toggle the goal-position debug marker."""
    if debug_vis:
        if not hasattr(env, "goal_pos_visualizer"):
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_edge_length = getattr(env.cfg, "touch_marker_edge_length", None)
            if marker_edge_length is None:
                touch_diameter = env._get_touch_threshold() * 2.0
                marker_scale = 0.9
                marker_size = touch_diameter * marker_scale
            else:
                marker_size = float(marker_edge_length)
            marker_cfg.markers["cuboid"].size = (marker_size, marker_size, marker_size)
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            env.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
        env.goal_pos_visualizer.set_visibility(True)
    else:
        if hasattr(env, "goal_pos_visualizer"):
            env.goal_pos_visualizer.set_visibility(False)


def debug_vis_callback(env, event) -> None:
    """Update the goal-position marker on each visualization callback."""
    if hasattr(env, "goal_pos_visualizer"):
        env.goal_pos_visualizer.visualize(env._desired_pos_w)
