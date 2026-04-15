from __future__ import annotations

import torch


def compute_dones(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target-touch terminated/time_out flags while preserving the original shapes."""
    time_out = env.episode_length_buf >= env.max_episode_length - 1

    died_by_height = env._get_logical_root_z() < float(getattr(env.cfg, "died_height_threshold", 0.3))
    died_by_tilt = env._get_tilt_exceeded_mask()
    died = died_by_height | env._get_ground_contact_mask() | died_by_tilt
    distance_to_goal = torch.linalg.norm(env._desired_pos_w - env._robot.data.root_pos_w, dim=1)
    far_away = distance_to_goal > env.cfg.far_away_termination_distance
    touched = distance_to_goal <= env._get_touch_threshold()
    env._term_touched = touched
    terminated = died | far_away | (touched if env.cfg.terminate_on_touch else torch.zeros_like(touched))
    return terminated, time_out
