from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import matrix_from_quat

from .drone_env_target_touch_vehicle import DroneTargetTouchVehicleEnv
from .drone_env_target_touch_vehicle_moving_cfg import DroneTargetTouchVehicleMovingBaseEnvCfg
from .markers import CUBOID_MARKER_CFG, POSITION_GOAL_MARKER_CFG, SPHERE_MARKER_CFG, VisualizationMarkers


class DroneTargetTouchVehicleMovingEnv(DroneTargetTouchVehicleEnv):
    """Vehicle-Moving 訓練環境。

    設計目標：
    - 保留 Vehicle 系列基底行為（父類：DroneTargetTouchVehicleEnv）。
    - 在此類別內加入 moving target 動態更新邏輯。
    """

    cfg: DroneTargetTouchVehicleMovingBaseEnvCfg

    def __init__(self, cfg: DroneTargetTouchVehicleMovingBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._target_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_dir_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_base_speed_mps = torch.full((self.num_envs,), float(self.cfg.moving_target_speed), device=self.device)
        self._target_speed_mps = torch.full((self.num_envs,), float(self.cfg.moving_target_speed), device=self.device)
        self._target_heading_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self._target_turn_rate_rad_s = torch.zeros(self.num_envs, device=self.device)
        self._target_turn_decision_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_z_phase = torch.zeros(self.num_envs, device=self.device)
        # 每回合起始距離（重置後即時計算），供測試輸出用。
        self._episode_start_distance = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._episode_min_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._episode_min_height = torch.full((self.num_envs,), float("inf"), device=self.device)
        # 每回合速度統計（m/s）：供測試列印時轉成 km/h。
        self._episode_speed_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_speed_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_speed_max = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._moving_target_dist_curriculum_stage_idx = -1

    def _sample_target_speeds(self, count: int) -> torch.Tensor:
        fixed_speed_cfg = getattr(self.cfg, "test_fixed_target_speed", None)
        if fixed_speed_cfg is not None:
            return torch.full((count,), float(fixed_speed_cfg), device=self.device)
        speed_min_cfg = getattr(self.cfg, "moving_target_speed_min", None)
        speed_max_cfg = getattr(self.cfg, "moving_target_speed_max", None)
        if speed_min_cfg is None or speed_max_cfg is None:
            return torch.full((count,), float(self.cfg.moving_target_speed), device=self.device)
        speed_min = float(speed_min_cfg)
        speed_max = float(speed_max_cfg)
        if speed_min > speed_max:
            speed_min, speed_max = speed_max, speed_min
        if abs(speed_max - speed_min) < 1e-6:
            return torch.full((count,), speed_min, device=self.device)
        return torch.empty(count, device=self.device).uniform_(speed_min, speed_max)

    def _sample_heading_hold_steps(self, count: int) -> torch.Tensor:
        hold_min_s = max(float(getattr(self.cfg, "moving_target_heading_hold_min_s", 2.0)), self.step_dt)
        hold_max_s = max(float(getattr(self.cfg, "moving_target_heading_hold_max_s", hold_min_s)), hold_min_s)
        steps_min = max(1, int(round(hold_min_s / self.step_dt)))
        steps_max = max(steps_min, int(round(hold_max_s / self.step_dt)))
        return torch.randint(steps_min, steps_max + 1, (count,), device=self.device)

    def _sample_turn_decision_steps(self, count: int) -> torch.Tensor:
        hold_min_s = max(float(getattr(self.cfg, "moving_target_turn_decision_min_s", 2.0)), self.step_dt)
        hold_max_s = max(float(getattr(self.cfg, "moving_target_turn_decision_max_s", hold_min_s)), hold_min_s)
        steps_min = max(1, int(round(hold_min_s / self.step_dt)))
        steps_max = max(steps_min, int(round(hold_max_s / self.step_dt)))
        return torch.randint(steps_min, steps_max + 1, (count,), device=self.device)

    def _sample_turn_segment_steps(self, count: int) -> torch.Tensor:
        hold_min_s = max(float(getattr(self.cfg, "moving_target_turn_segment_min_s", 2.0)), self.step_dt)
        hold_max_s = max(float(getattr(self.cfg, "moving_target_turn_segment_max_s", hold_min_s)), hold_min_s)
        steps_min = max(1, int(round(hold_min_s / self.step_dt)))
        steps_max = max(steps_min, int(round(hold_max_s / self.step_dt)))
        return torch.randint(steps_min, steps_max + 1, (count,), device=self.device)

    def _rotate_dirs_xy(self, dirs_w: torch.Tensor, delta_yaw_rad: torch.Tensor) -> torch.Tensor:
        """在 XY 平面旋轉方向向量，保留 Z 分量。"""
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

    def _compute_scene_obstacle_avoidance(
        self, env_ids: torch.Tensor, proposed_dir_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """檢查前視路徑是否會撞上靜態建物 bbox，並回傳是否需要轉彎與轉向號誌。"""
        hit_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        turn_signs = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)

        if not bool(getattr(self.cfg, "scene_obstacle_avoidance_enabled", False)):
            return hit_mask, turn_signs

        boxes_xy_min_all = getattr(self, "_scene_obstacle_boxes_xy_min", None)
        boxes_xy_max_all = getattr(self, "_scene_obstacle_boxes_xy_max", None)
        boxes_xy_center_all = getattr(self, "_scene_obstacle_boxes_xy_center", None)
        if boxes_xy_min_all is None or boxes_xy_max_all is None or boxes_xy_center_all is None:
            return hit_mask, turn_signs

        lookahead_min = max(float(getattr(self.cfg, "scene_obstacle_lookahead_min_m", 25.0)), 0.0)
        lookahead_time_s = max(float(getattr(self.cfg, "scene_obstacle_lookahead_time_s", 2.0)), 0.0)
        eps = 1e-6

        for local_idx, env_id_tensor in enumerate(env_ids):
            env_id = int(env_id_tensor.item())
            boxes_xy_min = boxes_xy_min_all[env_id]
            boxes_xy_max = boxes_xy_max_all[env_id]
            boxes_xy_center = boxes_xy_center_all[env_id]
            if boxes_xy_min is None or boxes_xy_max is None or boxes_xy_center is None or boxes_xy_min.numel() == 0:
                continue

            start_xy = self._desired_pos_w[env_id, :2]
            dir_xy = proposed_dir_w[local_idx, :2]
            dir_norm = torch.linalg.norm(dir_xy)
            if dir_norm <= eps:
                continue
            dir_xy = dir_xy / dir_norm
            current_speed = max(float(self._target_speed_mps[env_id].item()), 0.0)
            lookahead = max(lookahead_min, current_speed * lookahead_time_s)
            delta_xy = dir_xy * lookahead

            parallel_x = abs(float(delta_xy[0].item())) <= eps
            parallel_y = abs(float(delta_xy[1].item())) <= eps

            if parallel_x:
                tmin_x = torch.full((boxes_xy_min.shape[0],), float("-inf"), device=self.device)
                tmax_x = torch.full((boxes_xy_min.shape[0],), float("inf"), device=self.device)
                invalid_x = (start_xy[0] < boxes_xy_min[:, 0]) | (start_xy[0] > boxes_xy_max[:, 0])
            else:
                tx0 = (boxes_xy_min[:, 0] - start_xy[0]) / delta_xy[0]
                tx1 = (boxes_xy_max[:, 0] - start_xy[0]) / delta_xy[0]
                tmin_x = torch.minimum(tx0, tx1)
                tmax_x = torch.maximum(tx0, tx1)
                invalid_x = torch.zeros_like(tmin_x, dtype=torch.bool)

            if parallel_y:
                tmin_y = torch.full((boxes_xy_min.shape[0],), float("-inf"), device=self.device)
                tmax_y = torch.full((boxes_xy_min.shape[0],), float("inf"), device=self.device)
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

    def _apply_scene_obstacle_avoidance(
        self, env_ids: torch.Tensor, proposed_dir_w: torch.Tensor, motion_mode: str
    ) -> torch.Tensor:
        """若前方路徑會撞上建物，沿既有轉彎規則提早偏轉方向。"""
        hit_mask, turn_signs = self._compute_scene_obstacle_avoidance(env_ids, proposed_dir_w)
        if not torch.any(hit_mask):
            return proposed_dir_w

        adjusted_dir_w = proposed_dir_w.clone()
        hit_env_ids = env_ids[hit_mask]
        hit_count = int(hit_mask.sum().item())
        min_turn_rate = max(float(getattr(self.cfg, "moving_target_turn_rate_min_deg_s", 8.0)), 0.0)
        max_turn_rate = max(float(getattr(self.cfg, "moving_target_turn_rate_max_deg_s", min_turn_rate)), min_turn_rate)
        turn_rate_deg = torch.empty(hit_count, device=self.device).uniform_(min_turn_rate, max_turn_rate)
        turn_rate_rad = torch.deg2rad(turn_rate_deg) * turn_signs[hit_mask]
        adjusted_dir_w[hit_mask] = self._rotate_dirs_xy(adjusted_dir_w[hit_mask], turn_rate_rad * self.step_dt)

        if motion_mode == "road_like":
            self._target_turn_rate_rad_s[hit_env_ids] = turn_rate_rad
            self._target_turn_decision_steps[hit_env_ids] = self._sample_turn_segment_steps(hit_count)
            turn_speed_ratio = max(float(getattr(self.cfg, "moving_target_turn_speed_ratio", 0.75)), 0.0)
            self._target_speed_mps[hit_env_ids] = self._target_base_speed_mps[hit_env_ids] * turn_speed_ratio
            self._target_yaw_w[hit_env_ids] = torch.atan2(adjusted_dir_w[hit_mask, 1], adjusted_dir_w[hit_mask, 0])

        return adjusted_dir_w

    def _push_targets_out_of_scene_obstacles(self, env_ids: torch.Tensor):
        """若一步內仍踏入建物 bbox，就把目標位置推回 bbox 外。"""
        boxes_xy_min_all = getattr(self, "_scene_obstacle_boxes_xy_min", None)
        boxes_xy_max_all = getattr(self, "_scene_obstacle_boxes_xy_max", None)
        if boxes_xy_min_all is None or boxes_xy_max_all is None:
            return

        push_margin = max(float(getattr(self.cfg, "scene_obstacle_pushout_margin_m", 1.0)), 0.0)
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            boxes_xy_min = boxes_xy_min_all[env_id]
            boxes_xy_max = boxes_xy_max_all[env_id]
            if boxes_xy_min is None or boxes_xy_max is None or boxes_xy_min.numel() == 0:
                continue

            # 疊在多個 bbox 上時，最多推幾次，避免只推出一層又卡進另一層。
            for _ in range(4):
                point_xy = self._desired_pos_w[env_id, :2]
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
                    self._desired_pos_w[env_id, 0] = chosen_box_min[0] - push_margin
                elif chosen_face == 1:
                    self._desired_pos_w[env_id, 0] = chosen_box_max[0] + push_margin
                elif chosen_face == 2:
                    self._desired_pos_w[env_id, 1] = chosen_box_min[1] - push_margin
                else:
                    self._desired_pos_w[env_id, 1] = chosen_box_max[1] + push_margin

    def _update_moving_targets(self, env_ids: torch.Tensor):
        prev_dir_w = self._target_dir_w[env_ids]
        prev_norm = torch.linalg.norm(prev_dir_w, dim=1, keepdim=True)
        first_update = prev_norm.squeeze(-1) < 1e-6
        safe_prev_dir_w = prev_dir_w / torch.clamp(prev_norm, min=1e-6)
        motion_mode = str(getattr(self.cfg, "moving_target_motion_mode", "evade_drone")).lower()

        if motion_mode == "road_like":
            decision_steps = self._target_turn_decision_steps[env_ids] - 1
            self._target_turn_decision_steps[env_ids] = decision_steps
            resample_mask = first_update | (decision_steps <= 0)
            if torch.any(resample_mask):
                resample_env_ids = env_ids[resample_mask]
                sample_count = int(resample_mask.sum().item())
                straight_prob = min(max(float(getattr(self.cfg, "moving_target_straight_prob", 0.6)), 0.0), 1.0)
                is_straight = torch.rand(sample_count, device=self.device) < straight_prob
                turn_rates = torch.zeros(sample_count, device=self.device)
                min_turn_rate = max(float(getattr(self.cfg, "moving_target_turn_rate_min_deg_s", 8.0)), 0.0)
                max_turn_rate = max(float(getattr(self.cfg, "moving_target_turn_rate_max_deg_s", min_turn_rate)), min_turn_rate)
                turning_mask = ~is_straight
                if torch.any(turning_mask):
                    turn_count = int(turning_mask.sum().item())
                    mags_deg = torch.empty(turn_count, device=self.device).uniform_(min_turn_rate, max_turn_rate)
                    mags_rad = torch.deg2rad(mags_deg)
                    signs = torch.where(
                        torch.rand(turn_count, device=self.device) < 0.5,
                        -torch.ones(turn_count, device=self.device),
                        torch.ones(turn_count, device=self.device),
                    )
                    turn_rates[turning_mask] = signs * mags_rad
                self._target_turn_rate_rad_s[resample_env_ids] = turn_rates
                decision_steps = self._sample_turn_decision_steps(sample_count)
                if torch.any(turning_mask):
                    decision_steps[turning_mask] = self._sample_turn_segment_steps(int(turning_mask.sum().item()))
                self._target_turn_decision_steps[resample_env_ids] = decision_steps
                turn_speed_ratio = float(getattr(self.cfg, "moving_target_turn_speed_ratio", 0.75))
                turn_speed_ratio = max(turn_speed_ratio, 0.0)
                self._target_speed_mps[resample_env_ids] = self._target_base_speed_mps[resample_env_ids]
                if torch.any(turning_mask):
                    turning_env_ids = resample_env_ids[turning_mask]
                    self._target_speed_mps[turning_env_ids] = self._target_base_speed_mps[turning_env_ids] * turn_speed_ratio

            self._target_yaw_w[env_ids] = self._target_yaw_w[env_ids] + self._target_turn_rate_rad_s[env_ids] * self.step_dt
            new_dir_w = torch.zeros(len(env_ids), 3, device=self.device)
            new_dir_w[:, 0] = torch.cos(self._target_yaw_w[env_ids])
            new_dir_w[:, 1] = torch.sin(self._target_yaw_w[env_ids])
        else:
            if motion_mode == "random_straight":
                desired_dir_w = safe_prev_dir_w.clone()
                hold_steps = self._target_heading_hold_steps[env_ids] - 1
                self._target_heading_hold_steps[env_ids] = hold_steps
                resample_mask = first_update | (hold_steps <= 0)
                if torch.any(resample_mask):
                    resample_env_ids = env_ids[resample_mask]
                    sample_count = int(resample_mask.sum().item())
                    random_dir_w = torch.zeros(sample_count, 3, device=self.device)
                    min_turn_deg = float(getattr(self.cfg, "moving_target_heading_min_turn_deg", 0.0))
                    max_turn_deg = float(getattr(self.cfg, "moving_target_heading_max_turn_deg", 180.0))
                    min_turn_deg = max(0.0, min(180.0, min_turn_deg))
                    max_turn_deg = max(0.0, min(180.0, max_turn_deg))
                    if min_turn_deg > max_turn_deg:
                        min_turn_deg, max_turn_deg = max_turn_deg, min_turn_deg
                    min_turn_rad = torch.deg2rad(torch.tensor(min_turn_deg, device=self.device))
                    max_turn_rad = torch.deg2rad(torch.tensor(max_turn_deg, device=self.device))
                    prev_sel = safe_prev_dir_w[resample_mask]
                    prev_xy = prev_sel[:, :2]
                    prev_xy_norm = torch.linalg.norm(prev_xy, dim=1, keepdim=True)
                    prev_valid = prev_xy_norm.squeeze(-1) > 1e-6
                    prev_xy_unit = prev_xy / torch.clamp(prev_xy_norm, min=1e-6)
                    base_yaw = torch.atan2(prev_xy_unit[:, 1], prev_xy_unit[:, 0])
                    if max_turn_rad <= 0.0:
                        delta_yaw = torch.zeros(sample_count, device=self.device)
                    elif min_turn_rad <= 0.0:
                        delta_yaw = (torch.rand(sample_count, device=self.device) * 2.0 - 1.0) * max_turn_rad
                    else:
                        signs = torch.where(
                            torch.rand(sample_count, device=self.device) < 0.5,
                            -torch.ones(sample_count, device=self.device),
                            torch.ones(sample_count, device=self.device),
                        )
                        mags = torch.empty(sample_count, device=self.device).uniform_(
                            float(min_turn_rad), float(max_turn_rad)
                        )
                        delta_yaw = signs * mags
                    new_yaw = base_yaw + delta_yaw
                    random_dir_w[:, 0] = torch.cos(new_yaw)
                    random_dir_w[:, 1] = torch.sin(new_yaw)
                    # 首次更新（沒有前一方向）時，改用全域隨機方向初始化。
                    if torch.any(~prev_valid):
                        fallback_count = int((~prev_valid).sum().item())
                        fallback_yaw = torch.rand(fallback_count, device=self.device) * 2.0 * torch.pi
                        random_dir_w[~prev_valid, 0] = torch.cos(fallback_yaw)
                        random_dir_w[~prev_valid, 1] = torch.sin(fallback_yaw)
                    desired_dir_w[resample_mask] = random_dir_w
                    self._target_heading_hold_steps[resample_env_ids] = self._sample_heading_hold_steps(sample_count)
                    self._target_speed_mps[resample_env_ids] = self._sample_target_speeds(sample_count)
            else:
                target_pos_w = self._desired_pos_w[env_ids]
                drone_pos_w = self._robot.data.root_pos_w[env_ids]
                desired_dir_w = target_pos_w - drone_pos_w
                desired_dir_w[:, 2] = desired_dir_w[:, 2] * self.cfg.moving_target_vertical_dir_scale
                desired_dir_w = desired_dir_w / torch.clamp(torch.linalg.norm(desired_dir_w, dim=1, keepdim=True), min=1e-6)
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

        new_dir_w = self._apply_scene_obstacle_avoidance(env_ids, new_dir_w, motion_mode)
        self._target_dir_w[env_ids] = new_dir_w
        self._target_velocity_w[env_ids] = new_dir_w * self._target_speed_mps[env_ids].unsqueeze(-1)
        if self.cfg.moving_target_z_wave_amplitude > 0.0:
            phase_rate = 2.0 * torch.pi / self.cfg.moving_target_z_wave_period_s
            self._target_z_phase[env_ids] = self._target_z_phase[env_ids] + phase_rate * self.step_dt
            self._target_velocity_w[env_ids, 2] += (
                torch.sin(self._target_z_phase[env_ids]) * self.cfg.moving_target_z_wave_amplitude
            )
        self._desired_pos_w[env_ids] = self._desired_pos_w[env_ids] + self._target_velocity_w[env_ids] * self.step_dt

        height_offset = self._get_visual_altitude_offset()
        moving_target_z_min = float(self.cfg.moving_target_z_min) + height_offset
        moving_target_z_max = float(self.cfg.moving_target_z_max) + height_offset
        below = self._desired_pos_w[env_ids, 2] < moving_target_z_min
        above = self._desired_pos_w[env_ids, 2] > moving_target_z_max
        if torch.any(below):
            self._desired_pos_w[env_ids[below], 2] = moving_target_z_min
            self._target_velocity_w[env_ids[below], 2] = torch.abs(self._target_velocity_w[env_ids[below], 2])
        if torch.any(above):
            self._desired_pos_w[env_ids[above], 2] = moving_target_z_max
            self._target_velocity_w[env_ids[above], 2] = -torch.abs(self._target_velocity_w[env_ids[above], 2])

    def _pre_physics_step(self, actions: torch.Tensor):
        self._update_moving_targets(self._robot._ALL_INDICES)
        speed_mps = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        self._episode_speed_sum += speed_mps
        self._episode_speed_steps += 1.0
        self._episode_speed_max = torch.maximum(self._episode_speed_max, speed_mps)
        root_z = self._get_logical_root_z()
        self._episode_min_height = torch.minimum(self._episode_min_height, root_z)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        self._episode_min_distance = torch.minimum(self._episode_min_distance, distance_to_goal)
        super()._pre_physics_step(actions)

    def _resolve_moving_target_dist_curriculum_range(self) -> tuple[float, float] | None:
        if not bool(getattr(self.cfg, "target_distance_curriculum_enabled", False)):
            return None

        mode = str(getattr(self.cfg, "target_distance_curriculum_mode", "")).lower()
        if mode not in ("vector_step", "vector_steps", "timestep", "timesteps"):
            return None

        stages_cfg = getattr(self.cfg, "target_distance_curriculum_stages", None)
        if not stages_cfg:
            return None

        stages = [(float(s[0]), float(s[1])) for s in stages_cfg]
        stage_count = len(stages)
        stage_idx = stage_count - 1

        end_steps_cfg = getattr(self.cfg, "target_distance_curriculum_stage_end_steps", None)
        if end_steps_cfg and len(end_steps_cfg) == stage_count - 1:
            end_steps = [max(0, int(x)) for x in end_steps_cfg]
            for i, end_step in enumerate(end_steps):
                if self._vector_step_count < end_step:
                    stage_idx = i
                    break

        min_dist, max_dist = stages[stage_idx]
        if min_dist > max_dist:
            min_dist, max_dist = max_dist, min_dist

        if stage_idx != self._moving_target_dist_curriculum_stage_idx:
            self._moving_target_dist_curriculum_stage_idx = stage_idx
            print(
                f"[CURRICULUM][VehicleMovingTargetDist][vector_steps] step={self._vector_step_count} "
                f"stage={stage_idx + 1}/{stage_count} range={min_dist:.1f}~{max_dist:.1f}m",
                flush=True,
            )
        return min_dist, max_dist

    def _reset_idx_impl(self, env_ids: torch.Tensor | None, spread_episode_resets: bool):
        original_min = getattr(self.cfg, "target_spawn_distance_min", None)
        original_max = getattr(self.cfg, "target_spawn_distance_max", None)
        original_enabled = bool(getattr(self.cfg, "target_distance_curriculum_enabled", False))
        try:
            stage_range = self._resolve_moving_target_dist_curriculum_range()
            if stage_range is not None:
                self.cfg.target_spawn_distance_min = float(stage_range[0])
                self.cfg.target_spawn_distance_max = float(stage_range[1])
                self.cfg.target_distance_curriculum_enabled = False
            super()._reset_idx_impl(env_ids, spread_episode_resets)
        finally:
            self.cfg.target_spawn_distance_min = original_min
            self.cfg.target_spawn_distance_max = original_max
            self.cfg.target_distance_curriculum_enabled = original_enabled

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._target_velocity_w[env_ids] = 0.0
        self._target_dir_w[env_ids] = 0.0
        self._target_base_speed_mps[env_ids] = self._sample_target_speeds(len(env_ids))
        self._target_speed_mps[env_ids] = self._target_base_speed_mps[env_ids]
        self._target_heading_hold_steps[env_ids] = self._sample_heading_hold_steps(len(env_ids))
        self._target_yaw_w[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi
        self._target_turn_rate_rad_s[env_ids] = 0.0
        self._target_turn_decision_steps[env_ids] = self._sample_turn_decision_steps(len(env_ids))
        self._target_z_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi
        if bool(getattr(self.cfg, "scene_obstacle_spawn_clearance_enabled", False)):
            self._push_targets_out_of_scene_obstacles(env_ids)
        # 記錄這一回合的起始目標距離（重置後的目標-無人機距離）。
        self._episode_start_distance[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        self._episode_min_distance[env_ids] = float("inf")
        self._episode_min_height[env_ids] = float("inf")
        self._episode_speed_sum[env_ids] = 0.0
        self._episode_speed_steps[env_ids] = 0.0
        self._episode_speed_max[env_ids] = 0.0


class DroneTargetTouchVehicleMovingTestEnv(DroneTargetTouchVehicleMovingEnv):
    """Vehicle-Moving 測試環境。"""

    cfg: DroneTargetTouchVehicleMovingBaseEnvCfg

    def __init__(self, cfg: DroneTargetTouchVehicleMovingBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._trail_sample_counter = 0
        self._trail_env_id = int(getattr(self.cfg, "test_trail_env_id", 0))
        self._trail_max_points = max(1, int(getattr(self.cfg, "test_trail_max_points", 180)))
        self._trail_sample_steps = max(1, int(getattr(self.cfg, "test_trail_sample_steps", 2)))
        self._target_trail_points = torch.empty((0, 3), dtype=torch.float, device=self.device)
        self._drone_trail_points = torch.empty((0, 3), dtype=torch.float, device=self.device)
        self._touch_reset_delay_steps = max(0, int(getattr(self.cfg, "test_touch_reset_delay_steps", 0)))
        self._touch_reset_countdown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_edge_length = getattr(self.cfg, "touch_marker_edge_length", None)
                if marker_edge_length is None:
                    touch_diameter = self._get_touch_threshold() * 2.0
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
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

            if bool(getattr(self.cfg, "drone_outline_enabled", False)):
                if not hasattr(self, "drone_outline_visualizer"):
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.prim_path = "/Visuals/Command/drone_outline"
                    marker_cfg.markers["sphere"].radius = float(getattr(self.cfg, "drone_outline_radius", 0.28))
                    marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(getattr(self.cfg, "drone_outline_color", (0.15, 0.95, 1.0))),
                        emissive_color=tuple(getattr(self.cfg, "drone_outline_emissive_color", (0.08, 0.38, 0.42))),
                        opacity=float(getattr(self.cfg, "drone_outline_opacity", 0.18)),
                        roughness=0.2,
                    )
                    self.drone_outline_visualizer = VisualizationMarkers(marker_cfg)
                self.drone_outline_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "drone_outline_visualizer"):
                self.drone_outline_visualizer.set_visibility(False)

        trail_enabled = bool(getattr(self.cfg, "test_trail_enabled", True))
        if debug_vis and trail_enabled:
            if not hasattr(self, "trail_visualizer"):
                marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/vehicle_moving_trails"
                trail_marker_radius = float(getattr(self.cfg, "test_trail_marker_radius", 0.035))
                marker_cfg.markers["target_far"].radius = trail_marker_radius
                marker_cfg.markers["target_near"].radius = trail_marker_radius
                marker_cfg.markers["target_invisible"].radius = trail_marker_radius
                self.trail_visualizer = VisualizationMarkers(marker_cfg)
            self.trail_visualizer.set_visibility(True)
        else:
            if hasattr(self, "trail_visualizer"):
                self.trail_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Debug callback may be triggered during teardown after scene/robot is released.
        if not hasattr(self, "scene") or not hasattr(self, "_robot"):
            return
        try:
            if hasattr(self, "goal_pos_visualizer"):
                touched = (
                    torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
                    <= self._get_touch_threshold()
                )
                display_touched = touched | (self._touch_reset_countdown > 0)
                marker_indices = display_touched.to(dtype=torch.int32)
                self.goal_pos_visualizer.visualize(self._desired_pos_w, marker_indices=marker_indices)
            # if hasattr(self, "drone_outline_visualizer"):
            #     # 以根節點位置追蹤半透明外殼，讓遠景展示時無人機輪廓更明顯。
            #     self.drone_outline_visualizer.visualize(self._robot.data.root_pos_w)
            if not bool(getattr(self.cfg, "test_trail_enabled", True)):
                return
            if not hasattr(self, "trail_visualizer"):
                return
            if self._trail_env_id < 0 or self._trail_env_id >= self.num_envs:
                return

            self._trail_sample_counter += 1
            if self._trail_sample_counter % self._trail_sample_steps == 0:
                self._append_trail_sample(self._trail_env_id)

            target_count = int(self._target_trail_points.shape[0])
            drone_count = int(self._drone_trail_points.shape[0])
            total_count = target_count + drone_count
            if total_count == 0:
                self.trail_visualizer.set_visibility(False)
                return

            self.trail_visualizer.set_visibility(True)
            translations = torch.cat([self._target_trail_points, self._drone_trail_points], dim=0)
            marker_indices = torch.cat(
                [
                    torch.zeros(target_count, dtype=torch.int32, device=self.device),
                    torch.ones(drone_count, dtype=torch.int32, device=self.device),
                ],
                dim=0,
            )
            self.trail_visualizer.visualize(translations=translations, marker_indices=marker_indices)
        except Exception as exc:
            # Isaac Sim teardown / window close can invalidate PhysX tensor views before the callback is removed.
            if "invalidated" in str(exc).lower() or "backend" in str(exc).lower():
                return
            raise

    def _append_trail_sample(self, env_id: int):
        env_idx = int(env_id)
        target_point = self._desired_pos_w[env_idx : env_idx + 1].detach().clone()
        drone_back_point = self._compute_drone_back_point(env_idx).unsqueeze(0)
        self._target_trail_points = self._append_point_fifo(self._target_trail_points, target_point)
        self._drone_trail_points = self._append_point_fifo(self._drone_trail_points, drone_back_point)

    def _append_point_fifo(self, history: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        if history.shape[0] >= self._trail_max_points:
            history = history[-(self._trail_max_points - 1) :]
        return torch.cat([history, point], dim=0)

    def _compute_drone_back_point(self, env_id: int) -> torch.Tensor:
        root_pos_w = self._robot.data.root_pos_w[env_id]
        root_quat_w = self._robot.data.root_quat_w[env_id : env_id + 1]
        rot_mat_wb = matrix_from_quat(root_quat_w).reshape(3, 3)
        forward_w = rot_mat_wb[:, 0]
        back_offset = float(getattr(self.cfg, "test_trail_drone_back_offset", 0.35))
        return root_pos_w - forward_w * back_offset

    def _clear_trails(self):
        self._trail_sample_counter = 0
        self._target_trail_points = torch.empty((0, 3), dtype=torch.float, device=self.device)
        self._drone_trail_points = torch.empty((0, 3), dtype=torch.float, device=self.device)
        if hasattr(self, "trail_visualizer"):
            self.trail_visualizer.set_visibility(False)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        died_by_height = self._get_logical_root_z() < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_by_tilt = self._get_tilt_exceeded_mask()
        died = died_by_height | self._get_ground_contact_mask() | died_by_tilt
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        far_away = distance_to_goal > self.cfg.far_away_termination_distance
        touched = distance_to_goal <= self._get_touch_threshold()

        if self._touch_reset_delay_steps > 0 and self.cfg.terminate_on_touch:
            newly_touched = touched & (self._touch_reset_countdown == 0)
            self._touch_reset_countdown[newly_touched] = self._touch_reset_delay_steps
            touch_terminated = self._touch_reset_countdown == 1
            active_countdown = self._touch_reset_countdown > 0
            self._touch_reset_countdown[active_countdown] -= 1
            self._term_touched = touched | active_countdown | touch_terminated
        else:
            touch_terminated = touched if self.cfg.terminate_on_touch else torch.zeros_like(touched)
            self._term_touched = touched

        terminated = died | far_away | touch_terminated
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if not hasattr(self, "_test_reset_counter"):
            self._test_reset_counter = 0
            self._test_total_episodes = 0
            self._test_total_success = 0
            self._test_total_success_steps = 0.0
            self._test_total_died = 0
            self._test_total_dh = 0
            self._test_total_dg = 0
            self._test_total_dt = 0
            self._test_total_fa = 0
            self._test_total_to = 0
            self._test_total_fn = 0
            print("[TEST][VehicleMovingTouch] reset hook active", flush=True)
        self._test_reset_counter += 1

        self.cfg.curriculum_enabled = False
        self.cfg.died_height_threshold = 0.1

        if env_ids is None or len(env_ids) == self.num_envs:
            test_env_ids = self._robot._ALL_INDICES
        else:
            test_env_ids = env_ids

        self._touch_reset_countdown[test_env_ids] = 0

        if torch.any(test_env_ids == self._trail_env_id):
            self._clear_trails()

        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[test_env_ids] - self._robot.data.root_pos_w[test_env_ids], dim=1
        )
        died_by_height = self._get_logical_root_z(test_env_ids) < float(getattr(self.cfg, "died_height_threshold", 0.3))
        died_by_contact = self._get_ground_contact_mask(test_env_ids)
        died_by_tilt = self._get_tilt_exceeded_mask(test_env_ids)
        died_mask = died_by_height | died_by_contact | died_by_tilt
        far_mask = distance_to_goal > self.cfg.far_away_termination_distance
        time_out_mask = self.reset_time_outs[test_env_ids]
        touched_mask = self._term_touched[test_env_ids]
        fail_no_touch_mask = time_out_mask & (~touched_mask) & (~died_mask) & (~far_mask)
        touched_count = int(touched_mask.sum().item())
        min_dist_batch = self._episode_min_distance[test_env_ids]
        min_dist_summary = f"closest_dist={float(min_dist_batch.mean().item()):.3f}"
        min_height_batch = self._episode_min_height[test_env_ids]
        finite_min_height_mask = torch.isfinite(min_height_batch)
        if torch.any(finite_min_height_mask):
            min_height_values = min_height_batch[finite_min_height_mask]
            min_height_summary = f"lowest_z={float(min_height_values.mean().item()):.3f}"
        else:
            min_height_summary = "lowest_z=nan"
        start_dist_batch = self._episode_start_distance[test_env_ids]
        finite_start_mask = torch.isfinite(start_dist_batch)
        if torch.any(finite_start_mask):
            start_dist_values = start_dist_batch[finite_start_mask]
            start_dist_summary = f"start_dist={float(start_dist_values.mean().item()):.3f}"
        else:
            start_dist_summary = "start_dist=nan"
        speed_steps_batch = self._episode_speed_steps[test_env_ids]
        valid_speed_mask = speed_steps_batch > 0.0
        if torch.any(valid_speed_mask):
            speed_avg_batch = self._episode_speed_sum[test_env_ids][valid_speed_mask] / speed_steps_batch[valid_speed_mask]
            speed_max_batch = self._episode_speed_max[test_env_ids][valid_speed_mask]
            speed_summary = (
                f"speed_avg={float(speed_avg_batch.mean().item()):.3f}mps"
                f"({float((speed_avg_batch.mean() * 3.6).item()):.2f}kmh),"
                f"speed_max={float(speed_max_batch.mean().item()):.3f}mps"
                f"({float((speed_max_batch.mean() * 3.6).item()):.2f}kmh)"
            )
        else:
            speed_summary = "speed_avg=nan,speed_max=nan"
        reason_summary = (
            f"dh:{int(died_by_height.sum().item())},"
            f"dg:{int(died_by_contact.sum().item())},"
            f"dt:{int(died_by_tilt.sum().item())},"
            f"fa:{int(far_mask.sum().item())},"
            f"to:{int(time_out_mask.sum().item())},"
            f"fn:{int(fail_no_touch_mask.sum().item())}"
        )
        batch_total_reward = torch.zeros(len(test_env_ids), dtype=torch.float, device=self.device)
        for key in self._episode_sums.keys():
            batch_total_reward += self._episode_sums[key][test_env_ids]
        reward_summary = f"reward={float(batch_total_reward.mean().item()):.2f}"
        batch_episodes = int(touched_mask.numel())
        batch_dh = int(died_by_height.sum().item())
        batch_dg = int(died_by_contact.sum().item())
        batch_dt = int(died_by_tilt.sum().item())
        batch_fa = int(far_mask.sum().item())
        batch_to = int(time_out_mask.sum().item())
        batch_fn = int(fail_no_touch_mask.sum().item())
        batch_died = int(died_mask.sum().item())
        self._test_total_episodes += batch_episodes
        self._test_total_success += touched_count
        self._test_total_died += batch_died
        self._test_total_dh += batch_dh
        self._test_total_dg += batch_dg
        self._test_total_dt += batch_dt
        self._test_total_fa += batch_fa
        self._test_total_to += batch_to
        self._test_total_fn += batch_fn
        total_success_rate = (self._test_total_success / self._test_total_episodes) if self._test_total_episodes > 0 else 0.0
        if touched_count > 0:
            touched_steps = self.episode_length_buf[test_env_ids][touched_mask].float()
            self._test_total_success_steps += float(touched_steps.sum().item())
            total_avg_steps = self._test_total_success_steps / max(self._test_total_success, 1)
            print(
                "[TEST][VehicleMovingTouch] count="
                f"{touched_count}, "
                f"max_steps={float(touched_steps.max().item()):.1f}, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_steps={total_avg_steps:.1f}, "
                f"{reward_summary}, "
                f"{start_dist_summary}, "
                f"{min_dist_summary}, "
                f"{min_height_summary}, "
                f"{speed_summary}, "
                f"rsn={reason_summary}",
                flush=True,
            )
        else:
            print(
                "[TEST][VehicleMovingTouch] count=0, "
                f"total_rate={total_success_rate:.3f}, "
                f"total_avg_steps={self._test_total_success_steps / max(self._test_total_success, 1):.1f}, "
                f"{reward_summary}, "
                f"{start_dist_summary}, "
                f"{min_dist_summary}, "
                f"{min_height_summary}, "
                f"{speed_summary}, "
                f"rsn={reason_summary}",
                flush=True,
            )

        if self._test_reset_counter % 100 == 0:
            total_episodes = max(int(self._test_total_episodes), 1)
            total_died = int(self._test_total_died)
            total_died_den = max(total_died, 1)
            total_avg_steps = self._test_total_success_steps / max(self._test_total_success, 1)
            print(
                "[TEST][VehicleMovingTouch][TOTAL@"
                f"{self._test_reset_counter}] "
                f"episodes={self._test_total_episodes}, "
                f"success={self._test_total_success}, "
                f"pass_rate={self._test_total_success / total_episodes:.3f}, "
                f"avg_success_steps={total_avg_steps:.1f}, "
                f"died_total={total_died}, "
                f"died_rate={total_died / total_episodes:.3f}, "
                f"dh={self._test_total_dh}({self._test_total_dh / total_died_den:.3f}), "
                f"dg={self._test_total_dg}({self._test_total_dg / total_died_den:.3f}), "
                f"dt={self._test_total_dt}({self._test_total_dt / total_died_den:.3f}), "
                f"fa={self._test_total_fa}({self._test_total_fa / total_episodes:.3f}), "
                f"to={self._test_total_to}({self._test_total_to / total_episodes:.3f}), "
                f"fn={self._test_total_fn}({self._test_total_fn / total_episodes:.3f})",
                flush=True,
            )

        self._reset_idx_impl(env_ids, spread_episode_resets=True)
        if torch.any(test_env_ids == self._trail_env_id):
            self._append_trail_sample(self._trail_env_id)
