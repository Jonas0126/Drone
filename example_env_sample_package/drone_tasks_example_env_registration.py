"""Example Env task registration sample."""

import gymnasium as gym

from . import agents


gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Example-Env-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleExampleEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_example_env_cfg.yaml",
    },
)


gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Example-Env-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleExampleEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_example_env_cfg.yaml",
    },
)
