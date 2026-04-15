from __future__ import annotations

import torch

from .motion_sampling import rotate_dirs_xy, sample_turn_segment_steps


def compute_scene_obstacle_avoidance(
    env, env_ids: torch.Tensor, proposed_dir_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check whether the target lookahead path intersects cached scene obstacle boxes."""
    hit_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    turn_signs = torch.zeros(len(env_ids), dtype=torch.float, device=env.device)

    if not bool(getattr(env.cfg, "scene_obstacle_avoidance_enabled", False)):
        return hit_mask, turn_signs

    boxes_xy_min_all = getattr(env, "_scene_obstacle_boxes_xy_min", None)
    boxes_xy_max_all = getattr(env, "_scene_obstacle_boxes_xy_max", None)
    boxes_xy_center_all = getattr(env, "_scene_obstacle_boxes_xy_center", None)
    if boxes_xy_min_all is None or boxes_xy_max_all is None or boxes_xy_center_all is None:
        return hit_mask, turn_signs

    lookahead_min = max(float(getattr(env.cfg, "scene_obstacle_lookahead_min_m", 25.0)), 0.0)
    lookahead_time_s = max(float(getattr(env.cfg, "scene_obstacle_lookahead_time_s", 2.0)), 0.0)
    eps = 1e-6

    for local_idx, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        boxes_xy_min = boxes_xy_min_all[env_id]
        boxes_xy_max = boxes_xy_max_all[env_id]
        boxes_xy_center = boxes_xy_center_all[env_id]
        if boxes_xy_min is None or boxes_xy_max is None or boxes_xy_center is None or boxes_xy_min.numel() == 0:
            continue

        start_xy = env._desired_pos_w[env_id, :2]
        dir_xy = proposed_dir_w[local_idx, :2]
        dir_norm = torch.linalg.norm(dir_xy)
        if dir_norm <= eps:
            continue
        dir_xy = dir_xy / dir_norm
        current_speed = max(float(env._target_speed_mps[env_id].item()), 0.0)
        lookahead = max(lookahead_min, current_speed * lookahead_time_s)
        delta_xy = dir_xy * lookahead

        parallel_x = abs(float(delta_xy[0].item())) <= eps
        parallel_y = abs(float(delta_xy[1].item())) <= eps

        if parallel_x:
            tmin_x = torch.full((boxes_xy_min.shape[0],), float("-inf"), device=env.device)
            tmax_x = torch.full((boxes_xy_min.shape[0],), float("inf"), device=env.device)
            invalid_x = (start_xy[0] < boxes_xy_min[:, 0]) | (start_xy[0] > boxes_xy_max[:, 0])
        else:
            tx0 = (boxes_xy_min[:, 0] - start_xy[0]) / delta_xy[0]
            tx1 = (boxes_xy_max[:, 0] - start_xy[0]) / delta_xy[0]
            tmin_x = torch.minimum(tx0, tx1)
            tmax_x = torch.maximum(tx0, tx1)
            invalid_x = torch.zeros_like(tmin_x, dtype=torch.bool)

        if parallel_y:
            tmin_y = torch.full((boxes_xy_min.shape[0],), float("-inf"), device=env.device)
            tmax_y = torch.full((boxes_xy_min.shape[0],), float("inf"), device=env.device)
            invalid_y = (start_xy[1] < boxes_xy_min[:, 1]) | (start_xy[1] > boxes_xy_max[:, 1])
        else:
            ty0 = (boxes_xy_min[:, 1] - start_xy[1]) / delta_xy[1]
            ty1 = (boxes_xy_max[:, 1] - start_xy[1]) / delta_xy[1]
            tmin_y = torch.minimum(ty0, ty1)
            tmax_y = torch.maximum(ty0, ty1)
            invalid_y = torch.zeros_like(tmin_y, dtype=torch.bool)

        t_enter = torch.maximum(torch.maximum(tmin_x, tmin_y), torch.zeros_like(tmin_x))
        t_exit = torch.minimum(torch.minimum(tmax_x, tmax_y), torch.ones_like(tmax_x))
        hit_boxes = (~invalid_x) & (~invalid_y) & (t_exit >= t_enter)
        if not torch.any(hit_boxes):
            continue

        hit_indices = torch.nonzero(hit_boxes, as_tuple=False).squeeze(-1)
        nearest_local = torch.argmin(t_enter[hit_indices])
        box_idx = int(hit_indices[nearest_local].item())
        obstacle_center = boxes_xy_center[box_idx]
        to_center = obstacle_center - start_xy
        cross_z = dir_xy[0] * to_center[1] - dir_xy[1] * to_center[0]
        turn_signs[local_idx] = -1.0 if float(cross_z.item()) >= 0.0 else 1.0
        hit_mask[local_idx] = True

    return hit_mask, turn_signs


def apply_scene_obstacle_avoidance(env, env_ids: torch.Tensor, proposed_dir_w: torch.Tensor, motion_mode: str) -> torch.Tensor:
    """Bias the target direction away from upcoming scene obstacles."""
    hit_mask, turn_signs = compute_scene_obstacle_avoidance(env, env_ids, proposed_dir_w)
    if not torch.any(hit_mask):
        return proposed_dir_w

    adjusted_dir_w = proposed_dir_w.clone()
    hit_env_ids = env_ids[hit_mask]
    hit_count = int(hit_mask.sum().item())
    min_turn_rate = max(float(getattr(env.cfg, "moving_target_turn_rate_min_deg_s", 8.0)), 0.0)
    max_turn_rate = max(float(getattr(env.cfg, "moving_target_turn_rate_max_deg_s", min_turn_rate)), min_turn_rate)
    turn_rate_deg = torch.empty(hit_count, device=env.device).uniform_(min_turn_rate, max_turn_rate)
    turn_rate_rad = torch.deg2rad(turn_rate_deg) * turn_signs[hit_mask]
    adjusted_dir_w[hit_mask] = rotate_dirs_xy(adjusted_dir_w[hit_mask], turn_rate_rad * env.step_dt)

    if motion_mode == "road_like":
        env._target_turn_rate_rad_s[hit_env_ids] = turn_rate_rad
        env._target_turn_decision_steps[hit_env_ids] = sample_turn_segment_steps(env, hit_count)
        turn_speed_ratio = max(float(getattr(env.cfg, "moving_target_turn_speed_ratio", 0.75)), 0.0)
        env._target_speed_mps[hit_env_ids] = env._target_base_speed_mps[hit_env_ids] * turn_speed_ratio
        env._target_yaw_w[hit_env_ids] = torch.atan2(adjusted_dir_w[hit_mask, 1], adjusted_dir_w[hit_mask, 0])

    return adjusted_dir_w


def push_targets_out_of_scene_obstacles(env, env_ids: torch.Tensor) -> None:
    """Push targets outside cached obstacle boxes after spawn or motion."""
    boxes_xy_min_all = getattr(env, "_scene_obstacle_boxes_xy_min", None)
    boxes_xy_max_all = getattr(env, "_scene_obstacle_boxes_xy_max", None)
    if boxes_xy_min_all is None or boxes_xy_max_all is None:
        return

    push_margin = max(float(getattr(env.cfg, "scene_obstacle_pushout_margin_m", 1.0)), 0.0)
    for env_id_tensor in env_ids:
        env_id = int(env_id_tensor.item())
        boxes_xy_min = boxes_xy_min_all[env_id]
        boxes_xy_max = boxes_xy_max_all[env_id]
        if boxes_xy_min is None or boxes_xy_max is None or boxes_xy_min.numel() == 0:
            continue

        for _ in range(4):
            point_xy = env._desired_pos_w[env_id, :2]
            inside_mask = (
                (point_xy[0] >= boxes_xy_min[:, 0])
                & (point_xy[0] <= boxes_xy_max[:, 0])
                & (point_xy[1] >= boxes_xy_min[:, 1])
                & (point_xy[1] <= boxes_xy_max[:, 1])
            )
            if not torch.any(inside_mask):
                break

            inside_indices = torch.nonzero(inside_mask, as_tuple=False).squeeze(-1)
            exit_costs = []
            face_dists = []
            for idx in inside_indices:
                box_min = boxes_xy_min[idx]
                box_max = boxes_xy_max[idx]
                dist_to_faces = torch.stack(
                    [
                        point_xy[0] - box_min[0],
                        box_max[0] - point_xy[0],
                        point_xy[1] - box_min[1],
                        box_max[1] - point_xy[1],
                    ]
                )
                face_dists.append(dist_to_faces)
                exit_costs.append(torch.min(dist_to_faces))

            exit_costs_tensor = torch.stack(exit_costs)
            chosen_local = int(torch.argmin(exit_costs_tensor).item())
            chosen_box_idx = inside_indices[chosen_local]
            chosen_face_dists = face_dists[chosen_local]
            chosen_face = int(torch.argmin(chosen_face_dists).item())
            chosen_box_min = boxes_xy_min[chosen_box_idx]
            chosen_box_max = boxes_xy_max[chosen_box_idx]

            if chosen_face == 0:
                env._desired_pos_w[env_id, 0] = chosen_box_min[0] - push_margin
            elif chosen_face == 1:
                env._desired_pos_w[env_id, 0] = chosen_box_max[0] + push_margin
            elif chosen_face == 2:
                env._desired_pos_w[env_id, 1] = chosen_box_min[1] - push_margin
            else:
                env._desired_pos_w[env_id, 1] = chosen_box_max[1] + push_margin
