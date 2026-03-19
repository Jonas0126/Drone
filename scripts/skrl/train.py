# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["RICH_DISABLE"] = "1"
import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--checkpoint_weights_only",
    action="store_true",
    default=False,
    help="Load only model weights from checkpoint (skip optimizer/scheduler state).",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--debug_cam", action="store_true", default=False, help="Enable debug depth camera capture mode.")
parser.add_argument("--debug_collision", action="store_true", default=False, help="Enable debug collision mode.")
parser.add_argument(
    "--finetune",
    action="store_true",
    default=False,
    help="Enable fine-tuning mode (Stage1/Stage2/Stage3 only).",
)
parser.add_argument(
    "--reset-state-preprocessor",
    dest="reset_state_preprocessor",
    action="store_true",
    default=True,
    help="When --finetune is enabled, reset state preprocessor statistics (default: true).",
)
parser.add_argument(
    "--no-reset-state-preprocessor",
    dest="reset_state_preprocessor",
    action="store_false",
    help="When --finetune is enabled, keep loaded state preprocessor statistics.",
)
parser.add_argument(
    "--reset-value-preprocessor",
    dest="reset_value_preprocessor",
    action="store_true",
    default=True,
    help="When --finetune is enabled, reset value preprocessor statistics (default: true).",
)
parser.add_argument(
    "--no-reset-value-preprocessor",
    dest="reset_value_preprocessor",
    action="store_false",
    help="When --finetune is enabled, keep loaded value preprocessor statistics.",
)
parser.add_argument("--warmstart-lr", type=float, default=None, help="Override optimizer LR for fine-tune warm-start.")
parser.add_argument(
    "--warmstart-steps",
    type=int,
    default=0,
    help="Reserved for future warm-up schedule (currently only logged, no actor freeze).",
)
parser.add_argument(
    "--curriculum-level",
    type=int,
    default=None,
    choices=[0, 1, 2, 3, 4, 5],
    help="Obstacle curriculum level (0: none ... 5: five obstacles).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.debug_cam:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import random
import time
from datetime import datetime
from numbers import Number
from packaging import version

import skrl
import torch

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from utils.finetune import reset_preprocessors, set_optimizer_lr

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import Drone.tasks  # noqa: F401


def _print_optimizer_lrs(agent, tag: str):
    """Print current optimizer LR(s) for quick sanity check."""
    opts = None
    if hasattr(agent, "optimizers"):
        opts = agent.optimizers
    elif hasattr(agent, "optimizer"):
        opts = agent.optimizer

    if opts is None:
        print(f"[INFO] {tag} optimizer LR: <not found>")
        return

    if isinstance(opts, dict):
        opts = list(opts.values())
    elif not isinstance(opts, (list, tuple)):
        opts = [opts]

    lrs = []
    for opt in opts:
        if opt is None:
            continue
        for pg in getattr(opt, "param_groups", []):
            lr = pg.get("lr", None)
            if lr is not None:
                lrs.append(float(lr))

    if lrs:
        print(f"[INFO] {tag} optimizer LR(s): {lrs}")
    else:
        print(f"[INFO] {tag} optimizer LR: <not found>")


def _load_checkpoint_weights_only(agent, path: str):
    """Load checkpoint model weights only (no optimizer/scheduler state)."""
    if version.parse(torch.__version__) >= version.parse("1.13"):
        modules = torch.load(path, map_location=agent.device, weights_only=False)
    else:
        modules = torch.load(path, map_location=agent.device)

    if not isinstance(modules, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(modules)}")

    loaded_names: list[str] = []
    for name, model in agent.models.items():
        if model is None:
            continue
        state_dict = modules.get(name, None)
        if state_dict is None:
            print(f"[WARN] Weights-only load: missing model '{name}' in checkpoint")
            continue
        model.load_state_dict(state_dict)
        if hasattr(model, "eval"):
            model.eval()
        loaded_names.append(name)

    print(f"[INFO] Weights-only load complete. Loaded models: {loaded_names}")


class StepCounterVecEnvWrapper:
    """Count vectorized env steps and transitions for timestep interpretation debugging."""

    def __init__(self, env, print_interval: int = 1000):
        self.env = env
        self.print_interval = int(print_interval)
        self.vector_step_count = 0
        self.transition_count = 0
        self._agent = None

    def set_agent(self, agent):
        """Attach skrl agent for optional timestep introspection."""
        self._agent = agent

    def _infer_transition_batch_size(self, actions) -> int:
        if actions is None:
            return 0
        # For torch/jax arrays this corresponds to the vectorized env batch dimension.
        if hasattr(actions, "shape") and len(actions.shape) >= 1:
            try:
                return int(actions.shape[0])
            except Exception:
                pass
        try:
            return int(len(actions))
        except Exception:
            return 0

    def _get_skrl_timestep(self):
        if self._agent is None:
            return None
        # Different skrl versions/agents expose timestep counters using different attribute names.
        for attr in ("timestep", "_timestep", "timesteps", "_timesteps", "_step"):
            if not hasattr(self._agent, attr):
                continue
            value = getattr(self._agent, attr)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, Number):
                return int(value)
        return None

    def step(self, actions):
        out = self.env.step(actions)
        self.vector_step_count += 1
        self.transition_count += self._infer_transition_batch_size(actions)
        if self.print_interval > 0 and self.vector_step_count % self.print_interval == 0:
            skrl_timestep = self._get_skrl_timestep()
            ratio = (
                float(self.transition_count) / float(self.vector_step_count)
                if self.vector_step_count > 0
                else 0.0
            )
            print(
                "[TIMESTEP-DEBUG] "
                f"vector_step_count={self.vector_step_count} "
                f"transition_count={self.transition_count} "
                f"transitions_per_vector_step={ratio:.1f} "
                f"skrl_agent_timestep={skrl_timestep if skrl_timestep is not None else 'N/A'}",
                flush=True,
            )
        return out

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)



# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    if args_cli.finetune:
        allowed = bool(
            args_cli.task and (("Stage1" in args_cli.task) or ("Stage2" in args_cli.task) or ("Stage3" in args_cli.task))
        )
        if not allowed:
            raise ValueError("--finetune is restricted to Stage1/Stage2/Stage3 tasks only")
        if not args_cli.ml_framework.startswith("torch"):
            raise ValueError("--finetune currently supports only torch backend")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_cam = args_cli.debug_cam
    env_cfg.debug_collision = args_cli.debug_collision
    if args_cli.curriculum_level is not None and hasattr(env_cfg, "curriculum_level"):
        env_cfg.curriculum_level = args_cli.curriculum_level
        print(f"[INFO] Curriculum obstacle level: {env_cfg.curriculum_level}")

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # Stage0 timestep debug wrapper disabled.
    # (Uncomment the block below if timestep counting debug is needed again.)
    # is_stage0_task = bool(args_cli.task and "Stage0" in args_cli.task)
    # if is_stage0_task:
    #     env = StepCounterVecEnvWrapper(env, print_interval=1000)
    #     print("[INFO] Stage0 timestep debug counter enabled (print_interval=1000 vector steps).")

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)
    if isinstance(env, StepCounterVecEnvWrapper):
        env.set_agent(runner.agent)
    _print_optimizer_lrs(runner.agent, "Before checkpoint load")

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        if args_cli.checkpoint_weights_only:
            if not args_cli.ml_framework.startswith("torch"):
                print("[WARN] --checkpoint_weights_only only supports torch now. Fallback to full checkpoint load.")
                runner.agent.load(resume_path)
            else:
                _load_checkpoint_weights_only(runner.agent, resume_path)
        else:
            runner.agent.load(resume_path)
        _print_optimizer_lrs(runner.agent, "After checkpoint load")

    if args_cli.finetune:
        summary = reset_preprocessors(
            agent=runner.agent,
            env=env,
            device=env.unwrapped.device if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "device") else env.device,
            reset_state=args_cli.reset_state_preprocessor,
            reset_value=args_cli.reset_value_preprocessor,
        )
        print(
            "[FINETUNE] reset preprocessors: "
            f"state={args_cli.reset_state_preprocessor}, value={args_cli.reset_value_preprocessor}"
        )
        print(f"[FINETUNE] preprocessor summary: {summary}")

        if args_cli.warmstart_lr is not None:
            set_optimizer_lr(runner.agent, args_cli.warmstart_lr)
            print(f"[FINETUNE] warmstart_lr applied: {args_cli.warmstart_lr}")
            _print_optimizer_lrs(runner.agent, "After warmstart LR set")
        else:
            print("[FINETUNE] warmstart_lr not set; keeping checkpoint/config LR")

        print(f"[FINETUNE] warmstart_steps={args_cli.warmstart_steps} (reserved, no freeze applied)")

    # run training
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
