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
