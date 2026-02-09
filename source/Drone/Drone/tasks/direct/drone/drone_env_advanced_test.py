# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from pxr import Gf, UsdGeom

from .drone_env_advanced import DroneTrainEnv
from .drone_env_advanced_cfg import DroneTrainEnvCfg


class DroneTrainTestEnv(DroneTrainEnv):
    """Deterministic evaluation map for curriculum levels.

    - Fixed spawn/goal every episode
    - Obstacles are deterministically placed between spawn and goal
    - Obstacle count follows cfg.curriculum_level (0~5)
    """

    # Keep distance within max_goal_distance (15.0) to avoid instant too_far termination.
    _TEST_SPAWN_LOCAL = (2.0, 2.0, 2.0)
    _TEST_GOAL_LOCAL = (12.0, 12.0, 2.0)

    # Evenly spread on the path (center region) and alternate lateral offset
    _BARRIER_T_SEQ = (0.26, 0.38, 0.50, 0.64, 0.78)
    _BARRIER_OFFSETS = (0.0, 2.2, -2.2, 4.0, -4.0, 5.2, -5.2)
    _BARRIER_XY_CLEARANCE = 0.20

    def __init__(self, cfg: DroneTrainEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.debug_vis = True
        super().__init__(cfg, render_mode, **kwargs)

    def _pick_far_spawn_goal(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Fixed spawn/goal for stable and comparable testing."""
        return self._TEST_SPAWN_LOCAL, self._TEST_GOAL_LOCAL

    def _select_barrier_indices(self, total_blocks: int) -> list[int]:
        """Deterministic active block indices based on curriculum level."""
        count = max(0, min(int(getattr(self.cfg, "curriculum_level", 0)), self._BARRIER_BLOCKS))
        count = min(count, total_blocks)
        return list(range(count))

    def _place_barrier_blocks(
        self,
        env_id: int,
        spawn_local: tuple[float, float, float],
        goal_local: tuple[float, float, float],
    ) -> None:
        """Place blocks deterministically between spawn and goal, not too dense."""
        self._ensure_block_cache(env_id)
        blocks = self._blocks_per_env[env_id]
        if not blocks:
            return

        total_blocks = len(blocks)
        active_indices = self._select_barrier_indices(total_blocks)
        self._active_block_indices[env_id] = active_indices
        if len(active_indices) >= 3:
            self._tri_block_indices[env_id] = (active_indices[0], active_indices[1], active_indices[2])
        else:
            self._tri_block_indices[env_id] = None

        sx, sy, _ = spawn_local
        gx, gy, _ = goal_local
        dx, dy = gx - sx, gy - sy
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            dx, dy, norm = 1.0, 0.0, 1.0
        px, py = -dy / norm, dx / norm

        env_origin = self._terrain.env_origins[env_id]
        active_set = set(active_indices)
        placed_active: list[tuple[int, float, float]] = []

        for block_idx, block in enumerate(blocks):
            prim = block.GetPrim()
            prim.SetActive(True)

            xformable = UsdGeom.Xformable(block)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()

            if block_idx in active_set:
                order = active_indices.index(block_idx)
                t = self._BARRIER_T_SEQ[min(order, len(self._BARRIER_T_SEQ) - 1)]
                base = self._BARRIER_OFFSETS[min(order, len(self._BARRIER_OFFSETS) - 1)]

                # Try multiple lateral offsets and pick the first non-overlapping one.
                # This prevents small blocks from being buried inside large ones.
                candidates = [base] + [off for off in self._BARRIER_OFFSETS if off != base]
                found = False
                bx, by = 0.0, 0.0
                for lateral in candidates:
                    cx = sx + dx * t + px * lateral
                    cy = sy + dy * t + py * lateral
                    cx, cy = self._clamp_local_xy(float(cx), float(cy))
                    if all(
                        not self._blocks_overlap_xy_local(env_id, block_idx, cx, cy, prev_idx, px0, py0)
                        for (prev_idx, px0, py0) in placed_active
                    ):
                        bx, by = cx, cy
                        found = True
                        break

                # If no safe spot is found, hide this obstacle instead of allowing overlap.
                if found:
                    bz = self._SPAWN_Z
                    placed_active.append((block_idx, bx, by))
                else:
                    bx, by, bz = 0.0, 0.0, float(self.cfg.curriculum_hidden_z)
            else:
                bx, by, bz = 0.0, 0.0, float(self.cfg.curriculum_hidden_z)

            translate_op.Set(Gf.Vec3f(bx, by, bz))

            self._block_world_pos[(env_id, block_idx)] = Gf.Vec3d(
                float(env_origin[0] + bx),
                float(env_origin[1] + by),
                float(env_origin[2] + bz),
            )

    def _blocks_overlap_xy_local(
        self,
        env_id: int,
        block_a: int,
        ax: float,
        ay: float,
        block_b: int,
        bx: float,
        by: float,
    ) -> bool:
        """Check XY AABB overlap in local coordinates with a small clearance."""
        self._ensure_block_cache(env_id)
        if (
            env_id >= len(self._block_local_aabb)
            or block_a >= len(self._block_local_aabb[env_id])
            or block_b >= len(self._block_local_aabb[env_id])
        ):
            return False

        a_min_l, a_max_l = self._block_local_aabb[env_id][block_a]
        b_min_l, b_max_l = self._block_local_aabb[env_id][block_b]

        c = float(self._BARRIER_XY_CLEARANCE)
        a_min_x = ax + float(a_min_l[0]) - c
        a_max_x = ax + float(a_max_l[0]) + c
        a_min_y = ay + float(a_min_l[1]) - c
        a_max_y = ay + float(a_max_l[1]) + c

        b_min_x = bx + float(b_min_l[0]) - c
        b_max_x = bx + float(b_max_l[0]) + c
        b_min_y = by + float(b_min_l[1]) - c
        b_max_y = by + float(b_max_l[1]) + c

        return not (
            a_max_x < b_min_x
            or b_max_x < a_min_x
            or a_max_y < b_min_y
            or b_max_y < a_min_y
        )
