from __future__ import annotations

"""Compatibility facade for the vehicle-moving target-touch series."""

from .target_touch_vehicle_moving.env import (
    DroneTargetTouchVehicleMovingEnv,
    DroneTargetTouchVehicleMovingTestEnv,
)

__all__ = ["DroneTargetTouchVehicleMovingEnv", "DroneTargetTouchVehicleMovingTestEnv"]
