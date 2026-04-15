from __future__ import annotations

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms


def build_policy_observations(env) -> dict[str, torch.Tensor]:
    """Assemble the target-touch policy observations without changing tensor semantics."""
    if bool(getattr(env.cfg, "use_extended_observation", False)):
        rot_mat_wb = matrix_from_quat(env._robot.data.root_quat_w).reshape(env.num_envs, 9)
        root_pos_rel = env._robot.data.root_pos_w - env._terrain.env_origins
        desired_pos_rel = env._desired_pos_w - env._terrain.env_origins
        anchor_offset_xy = env._get_observation_anchor_offset_xy()
        if anchor_offset_xy is not None:
            root_pos_rel[:, :2] -= anchor_offset_xy
            desired_pos_rel[:, :2] -= anchor_offset_xy
        height_offset = env._get_visual_altitude_offset()
        if abs(height_offset) > 0.0:
            root_pos_rel[:, 2] -= height_offset
            desired_pos_rel[:, 2] -= height_offset
        obs = torch.cat(
            [
                root_pos_rel,
                env._robot.data.root_lin_vel_w,
                env._robot.data.root_ang_vel_w,
                rot_mat_wb,
                desired_pos_rel,
                env._actions,
            ],
            dim=-1,
        )
        return {"policy": obs}

    desired_pos_b, _ = subtract_frame_transforms(
        env._robot.data.root_pos_w,
        env._robot.data.root_quat_w,
        env._desired_pos_w,
    )
    obs = torch.cat(
        [
            env._robot.data.root_lin_vel_b,
            env._robot.data.root_ang_vel_b,
            env._robot.data.projected_gravity_b,
            desired_pos_b,
        ],
        dim=-1,
    )
    return {"policy": obs}
