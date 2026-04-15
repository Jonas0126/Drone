from __future__ import annotations

from ..target_touch.env import DroneTargetTouchEnv, DroneTargetTouchTestEnv
from .curriculum import vehicle_init, vehicle_pre_physics_step, vehicle_reset_idx_impl


class DroneTargetTouchVehicleEnv(DroneTargetTouchEnv):
    """Vehicle 系列專用 touch 環境（vector-step 課程版）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vehicle_init(self)

    def _pre_physics_step(self, actions):
        vehicle_pre_physics_step(self)
        super()._pre_physics_step(actions)

    def _reset_idx_impl(self, env_ids, spread_episode_resets: bool):
        vehicle_reset_idx_impl(self, env_ids, spread_episode_resets, super()._reset_idx_impl)


class DroneTargetTouchVehicleTestEnv(DroneTargetTouchTestEnv):
    """Vehicle 系列專用 touch 測試環境（vector-step 課程版）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vehicle_init(self)

    def _pre_physics_step(self, actions):
        vehicle_pre_physics_step(self)
        super()._pre_physics_step(actions)

    def _reset_idx_impl(self, env_ids, spread_episode_resets: bool):
        vehicle_reset_idx_impl(self, env_ids, spread_episode_resets, super()._reset_idx_impl)
