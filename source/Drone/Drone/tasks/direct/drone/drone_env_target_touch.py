from __future__ import annotations

"""Compatibility facade for the target-touch series.

The implementation now lives under ``target_touch/`` so the series can keep its own
modular layout without breaking existing imports and registry entry points.
"""

from .target_touch.env import DroneTargetTouchEnv, DroneTargetTouchTestEnv

__all__ = ["DroneTargetTouchEnv", "DroneTargetTouchTestEnv"]
