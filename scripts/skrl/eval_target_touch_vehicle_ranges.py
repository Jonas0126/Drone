# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate fixed-target tracking success rate over multiple respawn-distance ranges."""

import argparse
import random
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate target-touch checkpoint across distance ranges.")
parser.add_argument(
    "--task",
    type=str,
    default="Drone-Direct-Target-Touch-Vehicle-Stage2-v0",
    help="Evaluation task id (use non-Test env id).",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--episodes-per-stage", type=int, default=100, help="Episodes per distance stage.")
parser.add_argument(
    "--ranges",
    type=str,
    default="1-10,11-20,21-30,31-40,41-60",
    help="Comma-separated distance ranges in meters, e.g. '1-10,11-20,...'.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for evaluation.")
parser.add_argument("--max-steps-per-episode", type=int, default=4000, help="Safety cap for one episode.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from packaging import version
from tqdm.auto import tqdm

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


def _parse_ranges(ranges_text: str) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for token in ranges_text.split(","):
        token = token.strip()
        if not token:
            continue
        lo_hi = token.split("-")
        if len(lo_hi) != 2:
            raise ValueError(f"Invalid range token '{token}'. Expected format like '1-10'.")
        lo = float(lo_hi[0].strip())
        hi = float(lo_hi[1].strip())
        if lo > hi:
            lo, hi = hi, lo
        parsed.append((lo, hi))
    if not parsed:
        raise ValueError("No valid ranges parsed from --ranges.")
    return parsed


def _extract_success_from_info(info_obj, fallback_touched: bool) -> bool:
    # info may be dict or list-like depending on wrappers
    if isinstance(info_obj, dict):
        log = info_obj.get("log")
        if isinstance(log, dict):
            touched = log.get("Episode_Termination/touched")
            if touched is not None:
                try:
                    return float(touched) > 0.0
                except Exception:
                    pass
    return bool(fallback_touched)


def _extract_stage_flags(info_obj) -> dict:
    """Extract episode terminal flags from env info/log for num_envs=1 evaluation."""
    flags = {
        "died": 0,
        "time_out": 0,
        "touched": 0,
        "failed_no_touch": 0,
        "far_away": 0,
    }
    if not isinstance(info_obj, dict):
        return flags

    log = info_obj.get("log")
    if not isinstance(log, dict):
        return flags

    def _flag_from_log(key: str) -> int:
        value = log.get(key, 0.0)
        try:
            return int(float(value) > 0.0)
        except Exception:
            return 0

    flags["died"] = _flag_from_log("Episode_Termination/died")
    flags["time_out"] = _flag_from_log("Episode_Termination/time_out")
    flags["touched"] = _flag_from_log("Episode_Termination/touched")
    flags["failed_no_touch"] = _flag_from_log("Episode_Termination/failed_no_touch")
    # For num_envs=1, far_away_rate is either 0 or 1.
    flags["far_away"] = _flag_from_log("Metrics/far_away_rate")
    return flags


algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # Force single-drone evaluation.
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = True

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    stage_ranges = _parse_ranges(args_cli.ranges)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped

    wrapped_env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(wrapped_env, agent_cfg)
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")

    print("[EVAL] target-touch fixed-range evaluation starts")
    print(f"[EVAL] task={args_cli.task} checkpoint={checkpoint_path}")
    print(f"[EVAL] ranges={stage_ranges} episodes_per_stage={args_cli.episodes_per_stage}")

    total_success = 0
    total_episodes = 0
    total_touch_steps = 0
    total_failure_reasons = defaultdict(int)

    for stage_idx, (dist_min, dist_max) in enumerate(stage_ranges, start=1):
        # Keep spawn/reset logic identical to training env; only override target distance range.
        base_env.cfg.target_distance_curriculum_enabled = False
        base_env.cfg.target_spawn_distance_min = float(dist_min)
        base_env.cfg.target_spawn_distance_max = float(dist_max)

        obs, _ = wrapped_env.reset()
        stage_success = 0
        stage_episodes = 0
        stage_steps = 0
        stage_touch_steps = 0
        stage_failure_reasons = defaultdict(int)
        episode_steps = 0
        progress = tqdm(
            total=args_cli.episodes_per_stage,
            desc=f"Stage {stage_idx} [{dist_min:.1f}-{dist_max:.1f}m]",
            leave=True,
        )

        while simulation_app.is_running() and stage_episodes < args_cli.episodes_per_stage:
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(wrapped_env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in wrapped_env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, terminated, truncated, info = wrapped_env.step(actions)

            stage_steps += 1
            episode_steps += 1
            done_now = bool(terminated[0].item() or truncated[0].item())

            if done_now:
                # For this touch env, _term_touched reflects success on terminal step.
                touched_fallback = bool(getattr(base_env, "_term_touched")[0].item())
                succeeded = _extract_success_from_info(info, touched_fallback)
                flags = _extract_stage_flags(info)

                stage_success += int(succeeded)
                stage_episodes += 1
                total_success += int(succeeded)
                total_episodes += 1

                if succeeded:
                    stage_touch_steps += episode_steps
                    total_touch_steps += episode_steps
                else:
                    # Failure reason priority: died > far_away > failed_no_touch > time_out > unknown
                    if flags["died"]:
                        stage_failure_reasons["died"] += 1
                        total_failure_reasons["died"] += 1
                    elif flags["far_away"]:
                        stage_failure_reasons["far_away"] += 1
                        total_failure_reasons["far_away"] += 1
                    elif flags["failed_no_touch"]:
                        stage_failure_reasons["failed_no_touch"] += 1
                        total_failure_reasons["failed_no_touch"] += 1
                    elif flags["time_out"]:
                        stage_failure_reasons["time_out"] += 1
                        total_failure_reasons["time_out"] += 1
                    else:
                        stage_failure_reasons["unknown"] += 1
                        total_failure_reasons["unknown"] += 1

                episode_steps = 0
                progress.update(1)

            # Safety guard to avoid hanging episodes.
            if stage_steps > args_cli.max_steps_per_episode * args_cli.episodes_per_stage:
                print(
                    f"[WARN] Stage {stage_idx} exceeded safety step cap "
                    f"({args_cli.max_steps_per_episode * args_cli.episodes_per_stage})."
                )
                break

        progress.close()

        stage_rate = (stage_success / stage_episodes) if stage_episodes > 0 else 0.0
        stage_avg_touch_steps = (stage_touch_steps / stage_success) if stage_success > 0 else 0.0
        print(
            f"[RESULT][Stage {stage_idx}] range={dist_min:.1f}~{dist_max:.1f}m "
            f"success={stage_success}/{stage_episodes} success_rate={stage_rate:.3f} "
            f"avg_touch_steps={stage_avg_touch_steps:.1f}"
        )
        print(
            f"[RESULT][Stage {stage_idx}] failure_reasons="
            f"died:{stage_failure_reasons['died']},"
            f"far_away:{stage_failure_reasons['far_away']},"
            f"time_out:{stage_failure_reasons['time_out']},"
            f"failed_no_touch:{stage_failure_reasons['failed_no_touch']},"
            f"unknown:{stage_failure_reasons['unknown']}"
        )

    total_rate = (total_success / total_episodes) if total_episodes > 0 else 0.0
    total_avg_touch_steps = (total_touch_steps / total_success) if total_success > 0 else 0.0
    print("[RESULT][ALL] --------------------------------------------------")
    print(
        f"[RESULT][ALL] total_success={total_success}/{total_episodes} "
        f"total_success_rate={total_rate:.3f} avg_touch_steps={total_avg_touch_steps:.1f}"
    )
    print(
        f"[RESULT][ALL] failure_reasons="
        f"died:{total_failure_reasons['died']},"
        f"far_away:{total_failure_reasons['far_away']},"
        f"time_out:{total_failure_reasons['time_out']},"
        f"failed_no_touch:{total_failure_reasons['failed_no_touch']},"
        f"unknown:{total_failure_reasons['unknown']}"
    )
    if total_rate >= 0.8:
        print(f"[RESULT][ALL] PASS (total_success_rate={total_rate:.3f} >= 0.800)")
    else:
        print(f"[RESULT][ALL] FAIL (total_success_rate={total_rate:.3f} < 0.800)")
    print("[RESULT][ALL] --------------------------------------------------")

    wrapped_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
