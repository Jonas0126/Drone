# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Drone-Direct-Basic-v0",
    entry_point=f"{__name__}.drone_env_basic:DroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_basic_cfg:DroneEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Basic-Test-v0",
    entry_point=f"{__name__}.drone_env_basic_test:DroneTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_basic_cfg:DroneEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Advanced-v0",
    entry_point=f"{__name__}.drone_env_advanced:DroneTrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_advanced_cfg:DroneTrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_train_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Advanced-Test-v0",
    entry_point=f"{__name__}.drone_env_advanced_test:DroneTrainTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_advanced_cfg:DroneTrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_train_cfg.yaml",
        },
    )

# Fixed curriculum-level evaluation environments (0~5 obstacles)
for level in range(6):
    gym.register(
        id=f"Drone-Direct-Advanced-Level{level}-Test-v0",
        entry_point=f"{__name__}.drone_env_advanced_test:DroneTrainTestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.drone_env_advanced_cfg:DroneTrainLevel{level}EnvCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_train_cfg.yaml",
        },
    )

gym.register(
    id="Drone-Direct-Target-Touch-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_cfg:DroneTargetTouchEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_cfg:DroneTargetTouchEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Moving-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_cfg:DroneTargetTouchMovingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_moving_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Moving-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_cfg:DroneTargetTouchMovingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_moving_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Moving-Fast-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_fast_cfg:DroneTargetTouchMovingFastEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_moving_fast_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Moving-Fast-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_fast_cfg:DroneTargetTouchMovingFastEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_moving_fast_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage0-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage0EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage0_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage0-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage0EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage0_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage1-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage1_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage1-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage1_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage2-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage2EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage2_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage2-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage2EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage2_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage3-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage3_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage3-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage3_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage4-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage4EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage4_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage4-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage4EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage4_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage5-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage5EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage5_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Stage5-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch:DroneTargetTouchTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_vehicle_cfg:DroneTargetTouchVehicleStage5EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_stage5_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Faster-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_ladder_cfg:DroneTargetTouchMovingFasterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_faster_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-Faster-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_ladder_cfg:DroneTargetTouchMovingFasterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_faster_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-VeryFast-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_ladder_cfg:DroneTargetTouchMovingVeryFastEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_veryfast_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-VeryFast-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_ladder_cfg:DroneTargetTouchMovingVeryFastEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_veryfast_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-UltraFast-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_ladder_cfg:DroneTargetTouchMovingUltraFastEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_ultrafast_cfg.yaml",
    },
)

gym.register(
    id="Drone-Direct-Target-Touch-Vehicle-UltraFast-Test-v0",
    entry_point=f"{__name__}.drone_env_target_touch_moving:DroneTargetTouchMovingTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_target_touch_moving_ladder_cfg:DroneTargetTouchMovingUltraFastEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_target_touch_vehicle_ultrafast_cfg.yaml",
    },
)
