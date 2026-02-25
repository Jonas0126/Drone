from __future__ import annotations
# 中文說明：此檔案為無人機任務環境/設定實作，包含觀測、獎勵、終止與重置等核心邏輯。

import torch

from .drone_env_target_touch import DroneTargetTouchEnv
from .drone_env_target_touch_moving_cfg import DroneTargetTouchMovingEnvCfg


class DroneTargetTouchMovingEnv(DroneTargetTouchEnv):
    """移動目標版環境：目標會朝遠離無人機的方向移動。

    任務目標：
    - 在目標持續逃離時追上並碰觸目標。
    - reward 計算沿用 `DroneTargetTouchEnv._get_rewards`，
      只改變目標動態（target dynamics）。
    """

    cfg: DroneTargetTouchMovingEnvCfg

    def __init__(self, cfg: DroneTargetTouchMovingEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化移動目標環境狀態。"""
        super().__init__(cfg, render_mode, **kwargs)
        # 只作診斷用途：記錄目標目前世界座標速度。
        self._target_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        # 記錄前一時刻的移動方向，避免目標突然反向或急轉彎。
        self._target_dir_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_z_phase = torch.zeros(self.num_envs, device=self.device)

    def _update_moving_targets(self, env_ids: torch.Tensor):
        """更新指定環境的目標位置。

        規則：
        1. 以「目標位置 - 無人機位置」作為遠離方向。
        2. 依 `moving_target_speed` 前進。
        3. 對 Z 分量套用 `moving_target_vertical_dir_scale`，降低過大上下波動。
        4. 目標高度碰到上下界時做反射，避免越界。
        """
        target_pos_w = self._desired_pos_w[env_ids]
        drone_pos_w = self._robot.data.root_pos_w[env_ids]
        desired_dir_w = target_pos_w - drone_pos_w
        desired_dir_w[:, 2] = desired_dir_w[:, 2] * self.cfg.moving_target_vertical_dir_scale
        desired_dir_w = desired_dir_w / torch.clamp(torch.linalg.norm(desired_dir_w, dim=1, keepdim=True), min=1e-6)

        prev_dir_w = self._target_dir_w[env_ids]
        prev_norm = torch.linalg.norm(prev_dir_w, dim=1, keepdim=True)
        first_update = prev_norm.squeeze(-1) < 1e-6
        safe_prev_dir_w = prev_dir_w / torch.clamp(prev_norm, min=1e-6)
        safe_prev_dir_w[first_update] = desired_dir_w[first_update]

        if getattr(self.cfg, "moving_target_no_instant_reverse", True):
            dot_prev_desired = torch.sum(safe_prev_dir_w * desired_dir_w, dim=1, keepdim=True)
            reverse_mask = dot_prev_desired.squeeze(-1) < 0.0
            if torch.any(reverse_mask):
                adjusted_dir = desired_dir_w[reverse_mask] - dot_prev_desired[reverse_mask] * safe_prev_dir_w[reverse_mask]
                adjusted_norm = torch.linalg.norm(adjusted_dir, dim=1, keepdim=True)
                adjusted_dir = adjusted_dir / torch.clamp(adjusted_norm, min=1e-6)
                degenerate = adjusted_norm.squeeze(-1) < 1e-6
                if torch.any(degenerate):
                    adjusted_dir[degenerate] = safe_prev_dir_w[reverse_mask][degenerate]
                desired_dir_w[reverse_mask] = adjusted_dir

        turn_rate_limit = float(getattr(self.cfg, "moving_target_turn_rate_limit", 0.0))
        if turn_rate_limit > 0.0:
            max_turn_angle = turn_rate_limit * self.step_dt
            dot_prev_desired = torch.sum(safe_prev_dir_w * desired_dir_w, dim=1).clamp(-1.0, 1.0)
            turn_angle = torch.acos(dot_prev_desired)
            blend = torch.ones_like(turn_angle)
            over_limit = turn_angle > max_turn_angle
            blend[over_limit] = max_turn_angle / torch.clamp(turn_angle[over_limit], min=1e-6)
            new_dir_w = (1.0 - blend.unsqueeze(-1)) * safe_prev_dir_w + blend.unsqueeze(-1) * desired_dir_w
            new_dir_w = new_dir_w / torch.clamp(torch.linalg.norm(new_dir_w, dim=1, keepdim=True), min=1e-6)
            new_dir_w[first_update] = desired_dir_w[first_update]
        else:
            new_dir_w = desired_dir_w

        self._target_dir_w[env_ids] = new_dir_w
        self._target_velocity_w[env_ids] = new_dir_w * self.cfg.moving_target_speed
        if self.cfg.moving_target_z_wave_amplitude > 0.0:
            phase_rate = 2.0 * torch.pi / self.cfg.moving_target_z_wave_period_s
            self._target_z_phase[env_ids] = self._target_z_phase[env_ids] + phase_rate * self.step_dt
            self._target_velocity_w[env_ids, 2] += (
                torch.sin(self._target_z_phase[env_ids]) * self.cfg.moving_target_z_wave_amplitude
            )
        self._desired_pos_w[env_ids] = self._desired_pos_w[env_ids] + self._target_velocity_w[env_ids] * self.step_dt

        # 目標碰到 Z 邊界時做反射，避免鑽地或超出可用高度上限。
        below = self._desired_pos_w[env_ids, 2] < self.cfg.moving_target_z_min
        above = self._desired_pos_w[env_ids, 2] > self.cfg.moving_target_z_max
        if torch.any(below):
            self._desired_pos_w[env_ids[below], 2] = self.cfg.moving_target_z_min
            self._target_velocity_w[env_ids[below], 2] = torch.abs(self._target_velocity_w[env_ids[below], 2])
        if torch.any(above):
            self._desired_pos_w[env_ids[above], 2] = self.cfg.moving_target_z_max
            self._target_velocity_w[env_ids[above], 2] = -torch.abs(self._target_velocity_w[env_ids[above], 2])

    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步進前先更新移動目標，再套用動作。"""
        self._update_moving_targets(self._robot._ALL_INDICES)
        super()._pre_physics_step(actions)

    def _reset_idx_impl(self, env_ids: torch.Tensor | None, spread_episode_resets: bool):
        """重置時沿用父類流程，並清空目標速度。"""
        super()._reset_idx_impl(env_ids, spread_episode_resets)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._target_velocity_w[env_ids] = 0.0
        self._target_dir_w[env_ids] = 0.0
        self._target_z_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi


class DroneTargetTouchMovingTestEnv(DroneTargetTouchMovingEnv):
    """移動目標測試環境（固定重生條件，方便可重現比較）。

    固定內容：
    - 無人機起點固定在每個 env origin 的相對位置。
    - 目標初始點固定，之後依 moving 規則移動。
    """

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置測試環境，並輸出觸碰耗時統計。"""
        if not hasattr(self, "_test_reset_counter"):
            self._test_reset_counter = 0
            self._test_total_episodes = 0
            self._test_total_success = 0
            self._test_total_success_steps = 0.0
            print("[TEST][MovingTouch] DroneTargetTouchMovingTestEnv reset hook active", flush=True)
        self._test_reset_counter += 1
        # 測試模式：關閉課程，讓重生高度直接使用完整範圍。
        self.cfg.curriculum_enabled = False
        self.cfg.died_height_threshold = 0.1

        if env_ids is None or len(env_ids) == self.num_envs:
            test_env_ids = self._robot._ALL_INDICES
        else:
            test_env_ids = env_ids

        died_by_height = self._robot.data.root_pos_w[test_env_ids, 2] < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_by_contact = self._get_ground_contact_mask(test_env_ids)
        died_mask = died_by_height | died_by_contact
        touched_mask = self._term_touched[test_env_ids]
        touched_count = int(touched_mask.sum().item())
        died_count = int(died_mask.sum().item())
        if died_count > 0:
            died_heights = self._robot.data.root_pos_w[test_env_ids, 2][died_mask]
            died_height_values = ", ".join(f"{h:.3f}" for h in died_heights.tolist())
            reason_summary = f"died:{died_count}, died_heights:[{died_height_values}]"
        else:
            reason_summary = "died:0, died_heights:[]"
        batch_episodes = int(touched_mask.numel())
        self._test_total_episodes += batch_episodes
        self._test_total_success += touched_count
        total_success_rate = (self._test_total_success / self._test_total_episodes) if self._test_total_episodes > 0 else 0.0
        if touched_count > 0:
            touched_steps = self.episode_length_buf[test_env_ids][touched_mask].float()
            self._test_total_success_steps += float(touched_steps.sum().item())
            total_avg_steps = self._test_total_success_steps / max(self._test_total_success, 1)
            print(
                "[TEST][MovingTouch] count="
                f"{touched_count}, "
                f"max_steps={float(touched_steps.max().item()):.1f}, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_steps={total_avg_steps:.1f}, "
                f"{reason_summary}",
                flush=True,
            )
        else:
            print(
                "[TEST][MovingTouch] count=0, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_steps={self._test_total_success_steps / max(self._test_total_success, 1):.1f}, "
                f"{reason_summary}",
                flush=True,
            )

        self._reset_idx_impl(env_ids, spread_episode_resets=True)
