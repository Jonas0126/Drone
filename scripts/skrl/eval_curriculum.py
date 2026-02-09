# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a checkpoint on a fixed curriculum test environment."""

import argparse
import random
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate skrl checkpoint on curriculum test env.")
parser.add_argument("--task", type=str, required=True, help="Evaluation task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of evaluation environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for evaluation")
parser.add_argument("--max_steps", type=int, default=6000, help="Max environment steps for this evaluation run.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--curriculum-level", type=int, default=None, choices=[0, 1, 2, 3, 4, 5])

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from packaging import version

import skrl

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}")

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import Drone.tasks  # noqa: F401


algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = True

    if args_cli.curriculum_level is not None and hasattr(env_cfg, "curriculum_level"):
        env_cfg.curriculum_level = args_cli.curriculum_level

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(env, agent_cfg)
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")

    start_success = int(getattr(base_env, "_success_episodes_total", 0))
    obs, _ = env.reset()
    steps = 0

    while simulation_app.is_running() and steps < args_cli.max_steps:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
        steps += 1

    end_success = int(getattr(base_env, "_success_episodes_total", 0))
    success_episodes = max(0, end_success - start_success)
    print(f"EVAL_SUCCESS_EPISODES={success_episodes}")
    print(f"EVAL_STEPS={steps}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
