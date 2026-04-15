from __future__ import annotations

import torch

from ..target_touch_vehicle.env import DroneTargetTouchVehicleEnv
from .cfg import DroneTargetTouchVehicleMovingBaseEnvCfg
from .moving_target import init_moving_target_state, moving_pre_physics_step
from .reset_ops import moving_reset_idx_impl
from .test_mode import (
    compute_test_dones,
    init_test_mode_state,
    reset_test_idx,
    set_test_debug_vis,
    test_debug_vis_callback,
)


class DroneTargetTouchVehicleMovingEnv(DroneTargetTouchVehicleEnv):
    """Vehicle-Moving 訓練環境。

    設計目標：
    - 保留 Vehicle 系列基底行為（父類：DroneTargetTouchVehicleEnv）。
    - 在此類別內加入 moving target 動態更新邏輯。
    """

    cfg: DroneTargetTouchVehicleMovingBaseEnvCfg

    def __init__(self, cfg: DroneTargetTouchVehicleMovingBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Moving 系列自己的 episode/target 狀態統計，需在首次 reset 前就建立。
        init_moving_target_state(self)

    def _reset_idx_impl(self, env_ids: torch.Tensor | None, spread_episode_resets: bool):
        moving_reset_idx_impl(self, env_ids, spread_episode_resets, super()._reset_idx_impl)

    def _pre_physics_step(self, actions: torch.Tensor):
        moving_pre_physics_step(self)
        super()._pre_physics_step(actions)


class DroneTargetTouchVehicleMovingTestEnv(DroneTargetTouchVehicleMovingEnv):
    """Vehicle-Moving 測試環境。"""

    cfg: DroneTargetTouchVehicleMovingBaseEnvCfg

    def __init__(self, cfg: DroneTargetTouchVehicleMovingBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        init_test_mode_state(self)

    def _set_debug_vis_impl(self, debug_vis: bool):
        set_test_debug_vis(self, debug_vis)

    def _debug_vis_callback(self, event):
        test_debug_vis_callback(self, event)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_test_dones(self)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        reset_test_idx(self, env_ids)
