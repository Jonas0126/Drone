from __future__ import annotations

from .drone_env_target_touch import DroneTargetTouchEnv, DroneTargetTouchTestEnv


class _VehicleVectorCurriculumMixin:
    """Vehicle 系列專用：vector-step 驅動目標距離課程。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vector_step_count = 0
        self._vehicle_target_dist_stage_idx = -1

    def _pre_physics_step(self, actions):
        self._vector_step_count += 1
        super()._pre_physics_step(actions)

    def _resolve_vehicle_target_dist_range(self):
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

        if stage_idx != self._vehicle_target_dist_stage_idx:
            self._vehicle_target_dist_stage_idx = stage_idx
            print(
                f"[CURRICULUM][VehicleTargetDist][vector_steps] step={self._vector_step_count} "
                f"stage={stage_idx + 1}/{stage_count} range={min_dist:.1f}~{max_dist:.1f}m",
                flush=True,
            )

        return min_dist, max_dist

    def _reset_idx_impl(self, env_ids, spread_episode_resets: bool):
        # 暫時覆寫 target 距離範圍，讓父類別沿用原重生流程。
        original_min = getattr(self.cfg, "target_spawn_distance_min", None)
        original_max = getattr(self.cfg, "target_spawn_distance_max", None)
        original_enabled = bool(getattr(self.cfg, "target_distance_curriculum_enabled", False))

        try:
            stage_range = self._resolve_vehicle_target_dist_range()
            if stage_range is not None:
                self.cfg.target_spawn_distance_min = float(stage_range[0])
                self.cfg.target_spawn_distance_max = float(stage_range[1])
                # 關閉父類別 reset-step 課程分支，避免雙重課程。
                self.cfg.target_distance_curriculum_enabled = False

            super()._reset_idx_impl(env_ids, spread_episode_resets)
        finally:
            self.cfg.target_spawn_distance_min = original_min
            self.cfg.target_spawn_distance_max = original_max
            self.cfg.target_distance_curriculum_enabled = original_enabled


class DroneTargetTouchVehicleEnv(_VehicleVectorCurriculumMixin, DroneTargetTouchEnv):
    """Vehicle 系列專用 touch 環境（vector-step 課程版）。"""


class DroneTargetTouchVehicleTestEnv(_VehicleVectorCurriculumMixin, DroneTargetTouchTestEnv):
    """Vehicle 系列專用 touch 測試環境（vector-step 課程版）。"""
