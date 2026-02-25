# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

from pxr import Gf, UsdGeom

from .drone_env_basic import DroneEnv


class DroneTestEnv(DroneEnv):
    """測試用 Drone 環境：固定方塊與固定重生邏輯。"""

    # 固定 3 個方塊的位置（local 座標，會加上 env_origin）
    _FIXED_BLOCK_POSITIONS = (
        (1.0, 1.0, 2.0),
        (9.0, 9.0, 2.0),
        (5.0, 5.0, 2.0),
    )

    # 生成點偏移距離（沿著指向中間方塊的方向）
    _SPAWN_OFFSET = 2.0
    _SPAWN_CLEARANCE = 0.6

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """初始化測試環境並強制開啟目標點視覺化。

        Args:
            cfg: 環境設定物件。
            render_mode: 渲染模式（由 DirectRLEnv 解析）。
            **kwargs: 其他傳遞給父類的參數。
        """
        cfg.debug_vis = True
        super().__init__(cfg, render_mode, **kwargs)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定環境（固定方塊與固定 spawn/goal 生成）。

        Args:
            env_ids: 指定要重置的環境索引；None 表示全部重置。

        Returns:
            None
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # 回合統計紀錄
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = {
            "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
            "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
            "Metrics/final_distance_to_goal": final_distance_to_goal.item(),
        }
        self.extras["log"].update(extras)

        # 標準重置流程
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0

        if self.cfg.debug_collision:
            reset_env_ids = (
                set(env_ids.tolist()) if isinstance(env_ids, torch.Tensor) else set(int(e) for e in env_ids)
            )
            self._debug_collision_hits = {
                hit for hit in self._debug_collision_hits if hit[0] not in reset_env_ids
            }
            for env_id in reset_env_ids:
                self._debug_block_mesh_cache.pop(env_id, None)

        # 重置機體狀態
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # 目前環境原點
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # 重置方塊到固定位置，並記錄其座標
        self._reset_blocks_position(
            env_ids,
            x_range=(0, 10.0),
            y_range=(0, 10.0),
            z_range=(0.5, 8.0),
        )

        # 在最遠兩個方塊附近設置 spawn/goal，讓中間方塊形成阻隔
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        for batch_idx, env_id in enumerate(env_ids_list):
            tri = self._tri_block_indices[env_id]
            if tri is None:
                continue

            indices = list(tri)
            if len(indices) < 2:
                continue

            # 收集方塊世界座標
            positions = {idx: self._block_world_pos.get((env_id, idx)) for idx in indices}
            if any(p is None for p in positions.values()):
                continue

            # 找出最遠方塊配對
            far_a, far_b = indices[0], indices[1]
            far_dist = -1.0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pi = positions[indices[i]]
                    pj = positions[indices[j]]
                    d = math.dist((pi[0], pi[1], pi[2]), (pj[0], pj[1], pj[2]))
                    if d > far_dist:
                        far_dist = d
                        far_a, far_b = indices[i], indices[j]

            # 剩下的索引即中間方塊
            mid = next(idx for idx in indices if idx not in (far_a, far_b))

            pos_a = positions[far_a]
            pos_b = positions[far_b]
            pos_m = positions[mid]

            # 方向：靠向中間方塊，確保中間有障礙
            dir_a = self._normalize_xy((pos_m[0] - pos_a[0], pos_m[1] - pos_a[1]))
            dir_b = self._normalize_xy((pos_m[0] - pos_b[0], pos_m[1] - pos_b[1]))

            # 根據方塊 AABB 做最小安全偏移，避免 spawn/goal 掉進方塊內
            offset_a = self._compute_safe_offset(env_id, far_a, dir_a)
            offset_b = self._compute_safe_offset(env_id, far_b, dir_b)

            spawn_world = (
                pos_a[0] + dir_a[0] * offset_a,
                pos_a[1] + dir_a[1] * offset_a,
                pos_a[2],
            )
            goal_world = (
                pos_b[0] + dir_b[0] * offset_b,
                pos_b[1] + dir_b[1] * offset_b,
                pos_b[2],
            )

            spawn_pos = self._clamp_world_xy(spawn_world, env_id)
            goal_pos = self._clamp_world_xy(goal_world, env_id)

            # 最後保險：如果仍在任一 active block 內，沿方向再推遠一點
            spawn_pos = self._push_out_of_blocks(env_id, spawn_pos, dir_a, indices)
            goal_pos = self._push_out_of_blocks(env_id, goal_pos, dir_b, indices)

            default_root_state[batch_idx, 0:3] = torch.tensor(
                spawn_pos,
                device=default_root_state.device,
                dtype=default_root_state.dtype,
            )
            self._desired_pos_w[env_id, :] = torch.tensor(
                goal_pos,
                device=default_root_state.device,
                dtype=default_root_state.dtype,
            )

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        d = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        self._prev_distance[env_ids] = d
        self._hover_count[env_ids] = 0

    def _reset_blocks_position(
        self,
        env_ids,
        x_range=(0, 20.0),
        y_range=(0, 20.0),
        z_range=(0.5, 20.0),
    ):
        """固定 3 個方塊位置，其餘方塊隱藏到地面下。

        Args:
            env_ids: 要重設的環境索引（list 或 Tensor）。
            x_range: 位置範圍 (min, max)，僅作為邊界參考。
            y_range: 位置範圍 (min, max)，僅作為邊界參考。
            z_range: 位置範圍 (min, max)，僅作為邊界參考。

        Returns:
            None
        """
        for env_id in env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids:
            self._ensure_block_cache(int(env_id))
            env_id = int(env_id)

            blocks = self._blocks_per_env[env_id]
            if not blocks:
                continue

            total_blocks = len(blocks)
            tri_indices = list(range(min(3, total_blocks)))
            self._active_block_indices[env_id] = tri_indices
            if len(tri_indices) == 3:
                self._tri_block_indices[env_id] = (tri_indices[0], tri_indices[1], tri_indices[2])
            else:
                self._tri_block_indices[env_id] = None

            env_origin = self._terrain.env_origins[env_id]

            for block_idx, block in enumerate(blocks):
                prim = block.GetPrim()
                prim.SetActive(True)

                xformable = UsdGeom.Xformable(block)
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()

                if block_idx in tri_indices:
                    x, y, z = self._FIXED_BLOCK_POSITIONS[block_idx]
                else:
                    x, y, z = 0.0, 0.0, float(self.cfg.curriculum_hidden_z)

                translate_op.Set(Gf.Vec3f(x, y, z))

                # 同步 Python 端世界座標快取
                self._block_world_pos[(env_id, block_idx)] = Gf.Vec3d(
                    float(env_origin[0] + x),
                    float(env_origin[1] + y),
                    float(env_origin[2] + z),
                )

    def _clamp_world_xy(self, pos: tuple[float, float, float], env_id: int) -> tuple[float, float, float]:
        """將 world 位置限制在安全範圍內（避免超出場地）。

        Args:
            pos: (x, y, z) 的 world 座標。
            env_id: 環境索引。

        Returns:
            tuple[float, float, float]: 限制後的 world 座標位置。
        """
        env_origin = self._terrain.env_origins[env_id]
        x_min, x_max = float(env_origin[0] + 0.5), float(env_origin[0] + 9.5)
        y_min, y_max = float(env_origin[1] + 0.5), float(env_origin[1] + 9.5)

        x = min(max(pos[0], x_min), x_max)
        y = min(max(pos[1], y_min), y_max)
        z = pos[2]

        return (float(x), float(y), float(z))

    def _normalize_xy(self, vec: tuple[float, float]) -> tuple[float, float]:
        """正規化 2D 向量，避免除以 0。

        Args:
            vec: (x, y) 向量。

        Returns:
            tuple[float, float]: 正規化後的向量。
        """
        x, y = vec
        norm = math.sqrt(x * x + y * y)
        if norm < 1e-6:
            return 1.0, 0.0
        return x / norm, y / norm

    def _compute_safe_offset(self, env_id: int, block_idx: int, direction: tuple[float, float]) -> float:
        """根據方塊 AABB 計算安全偏移距離。

        Args:
            env_id: 環境索引。
            block_idx: 方塊索引。
            direction: 2D 方向向量（已正規化）。

        Returns:
            float: 安全偏移距離。
        """
        self._ensure_block_cache(env_id)
        if env_id >= len(self._block_local_aabb) or block_idx >= len(self._block_local_aabb[env_id]):
            return self._SPAWN_OFFSET

        local_min, local_max = self._block_local_aabb[env_id][block_idx]
        half_x = max(abs(local_min[0]), abs(local_max[0]))
        half_y = max(abs(local_min[1]), abs(local_max[1]))
        # 沿方向投影到 AABB 外側的最小距離
        proj = abs(direction[0]) * half_x + abs(direction[1]) * half_y
        return max(self._SPAWN_OFFSET, proj + self._SPAWN_CLEARANCE)

    def _push_out_of_blocks(
        self,
        env_id: int,
        pos: tuple[float, float, float],
        direction: tuple[float, float],
        block_indices: list[int],
    ) -> tuple[float, float, float]:
        """若位置落在任一方塊內，沿指定方向推出去。

        Args:
            env_id: 環境索引。
            pos: world 位置。
            direction: 2D 方向向量（已正規化）。
            block_indices: 要檢查的方塊索引。

        Returns:
            tuple[float, float, float]: 推出後的位置。
        """
        x, y, z = pos
        for _ in range(4):
            inside = False
            for idx in block_indices:
                aabb_min, aabb_max = self._compute_block_world_aabb(env_id, idx)
                if (
                    x >= aabb_min[0]
                    and x <= aabb_max[0]
                    and y >= aabb_min[1]
                    and y <= aabb_max[1]
                    and z >= aabb_min[2]
                    and z <= aabb_max[2]
                ):
                    inside = True
                    break
            if not inside:
                break
            x += direction[0] * self._SPAWN_CLEARANCE
            y += direction[1] * self._SPAWN_CLEARANCE
        return self._clamp_world_xy((x, y, z), env_id)
