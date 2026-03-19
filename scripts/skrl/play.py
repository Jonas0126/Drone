# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--image-sequence",
    "--image_sequence",
    dest="image_sequence",
    action="store_true",
    default=False,
    help="Record an Isaac Sim image sequence (PNG frames) with Replicator BasicWriter.",
)
parser.add_argument(
    "--image-sequence-dir",
    "--image_sequence_dir",
    dest="image_sequence_dir",
    type=str,
    default=None,
    help="Output directory for Isaac Sim image-sequence capture. Defaults to logs/.../image_sequences/play/<timestamp>.",
)
parser.add_argument(
    "--image-sequence-camera",
    "--image_sequence_camera",
    dest="image_sequence_camera",
    type=str,
    default="/OmniverseKit_Persp",
    help="Camera prim path used for image-sequence capture.",
)
parser.add_argument(
    "--image-sequence-rt-subframes",
    "--image_sequence_rt_subframes",
    dest="image_sequence_rt_subframes",
    type=int,
    default=0,
    help="Optional Replicator RT subframes for each captured image. Higher values can improve quality but slow capture.",
)
parser.add_argument(
    "--test_fixed_distance",
    type=float,
    default=None,
    help="For test envs, override target spawn distance to a fixed value in meters.",
)
parser.add_argument(
    "--test_fixed_speed",
    type=float,
    default=None,
    help="For moving test envs, override target speed to a fixed value in m/s.",
)
parser.add_argument(
    "--test_num_episodes",
    type=int,
    default=None,
    help="For test envs, stop playback after the given total number of completed episodes.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=None,
    help="Output video FPS. Defaults to environment step frequency for real-time playback speed.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--width",
    type=int,
    default=0,
    help="Render width for playback/recording. Use <= 0 to keep environment default.",
)
parser.add_argument(
    "--height",
    type=int,
    default=0,
    help="Render height for playback/recording. Use <= 0 to keep environment default.",
)
parser.add_argument(
    "--num_envs",
    "--env_nums",
    dest="num_envs",
    type=int,
    default=None,
    help="Number of environments to simulate.",
)
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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--render-real-time2",
    "--render_real_time2",
    dest="render_real_time2",
    action="store_true",
    default=False,
    help="Use Omniverse RTX Real-Time 2.0 renderer (RaytracedLighting).",
)
parser.add_argument("--debug_cam", action="store_true", default=False, help="Enable debug depth camera capture mode.")
parser.add_argument("--debug_collision", action="store_true", default=False, help="Enable debug collision mode.")
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="Enable chase camera that follows the drone during play/video.",
)
parser.add_argument(
    "--follow_env_id",
    type=int,
    default=0,
    help="Environment index to follow when --follow_camera is enabled.",
)
parser.add_argument(
    "--follow_distance",
    type=float,
    default=8.0,
    help="Chase camera distance behind the drone (meters).",
)
parser.add_argument(
    "--follow_height",
    type=float,
    default=3.0,
    help="Chase camera height offset above the drone (meters).",
)
parser.add_argument(
    "--follow_look_ahead",
    type=float,
    default=4.0,
    help="Look-ahead distance in front of the drone (meters).",
)
parser.add_argument(
    "--follow_subject",
    type=str,
    default="drone",
    choices=["drone", "target", "both"],
    help="Camera follow subject when --follow_camera is enabled.",
)
parser.add_argument(
    "--follow_smooth_alpha",
    type=float,
    default=0.2,
    help="Camera smoothing factor in [0, 1]. Set 0 to disable smoothing.",
)
parser.add_argument(
    "--follow_both_lock_distance",
    type=float,
    default=2.0,
    help="In `both` mode, lock view direction when drone-target distance is below this value (meters).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video and args_cli.image_sequence:
    parser.error("--video and --image-sequence cannot be used together. Choose one recording mode.")
# always enable cameras to record video
if args_cli.video or args_cli.image_sequence:
    args_cli.enable_cameras = True
if args_cli.debug_cam:
    args_cli.enable_cameras = True
if args_cli.render_real_time2:
    # Force Omniverse's RTX Real-Time 2.0 renderer without overloading the existing playback-speed flag.
    kit_arg = "--/rtx/rendermode=RaytracedLighting"
    existing_kit_args = getattr(args_cli, "kit_args", "") or ""
    if "/rtx/rendermode" not in existing_kit_args:
        args_cli.kit_args = f"{existing_kit_args} {kit_arg}".strip()
    print("[INFO] Forcing renderer mode: RTX Real-Time 2.0 (RaytracedLighting)")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb

carb_settings = carb.settings.get_settings()
# Keep Isaac Sim path-tracing bounce limits consistent with the requested demo render preferences.
carb_settings.set_int("/rtx/pathtracing/maxBounces", 6)
carb_settings.set_int("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
print(
    "[INFO] PathTracing bounce settings: "
    f"maxBounces={carb_settings.get('/rtx/pathtracing/maxBounces')}, "
    "maxSpecularAndTransmissionBounces="
    f"{carb_settings.get('/rtx/pathtracing/maxSpecularAndTransmissionBounces')}"
)

if args_cli.render_real_time2:
    # Kit args usually work, but explicitly setting the carb value after app startup is more reliable.
    carb_settings.set_string("/rtx/rendermode", "RaytracedLighting")
    print(f"[INFO] Active RTX render mode: {carb_settings.get('/rtx/rendermode')}")

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
from packaging import version

import skrl

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
from isaaclab.utils.dict import print_dict
import isaaclab.utils.math as math_utils
import omni.replicator.core as rep

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Drone.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = True
    env_cfg.debug_cam = args_cli.debug_cam
    env_cfg.debug_collision = args_cli.debug_collision
    if args_cli.test_fixed_distance is not None:
        env_cfg.target_spawn_distance_min = float(args_cli.test_fixed_distance)
        env_cfg.target_spawn_distance_max = float(args_cli.test_fixed_distance)
        if hasattr(env_cfg, "target_distance_curriculum_enabled"):
            env_cfg.target_distance_curriculum_enabled = False
    if args_cli.test_fixed_speed is not None and hasattr(env_cfg, "test_fixed_target_speed"):
        env_cfg.test_fixed_target_speed = float(args_cli.test_fixed_speed)

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    # Keep RGB recording resolution aligned with CLI width/height.
    if hasattr(env_cfg, "viewer") and env_cfg.viewer is not None:
        viewer_w, viewer_h = env_cfg.viewer.resolution
        if int(args_cli.width) > 0:
            viewer_w = int(args_cli.width)
        if int(args_cli.height) > 0:
            viewer_h = int(args_cli.height)
        env_cfg.viewer.resolution = (viewer_w, viewer_h)
        if args_cli.video:
            print(f"[INFO] Video capture resolution: {viewer_w}x{viewer_h}")
        if args_cli.image_sequence:
            print(f"[INFO] Image-sequence capture resolution: {viewer_w}x{viewer_h}")
    else:
        viewer_w = int(args_cli.width) if int(args_cli.width) > 0 else 1280
        viewer_h = int(args_cli.height) if int(args_cli.height) > 0 else 720

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    raw_env = env.unwrapped

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        # Default to environment control-step frequency so recorded playback matches simulated time scale.
        video_fps = int(round(1.0 / dt)) if args_cli.video_fps is None else max(1, int(args_cli.video_fps))
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "fps": video_fps,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    image_sequence_output_dir = None
    image_sequence_writer = None
    image_sequence_render_product = None
    image_sequence_capture_limit = int(args_cli.video_length)
    image_sequence_frame_count = 0
    if args_cli.image_sequence:
        image_sequence_output_dir = (
            os.path.abspath(args_cli.image_sequence_dir)
            if args_cli.image_sequence_dir
            else os.path.join(log_dir, "image_sequences", "play", time.strftime("%Y-%m-%d_%H-%M-%S"))
        )
        os.makedirs(image_sequence_output_dir, exist_ok=True)
        rep.orchestrator.set_capture_on_play(False)
        image_sequence_render_product = rep.create.render_product(
            args_cli.image_sequence_camera,
            (int(viewer_w), int(viewer_h)),
            force_new=True,
        )
        image_sequence_writer = rep.writers.get("BasicWriter")
        image_sequence_writer.initialize(
            output_dir=image_sequence_output_dir,
            rgb=True,
            frame_padding=6,
        )
        image_sequence_writer.attach([image_sequence_render_product])
        print(
            "[INFO] Recording Isaac Sim image sequence.",
            flush=True,
        )
        print(
            f"[INFO] Image-sequence output: {image_sequence_output_dir} "
            f"(camera={args_cli.image_sequence_camera}, frames={image_sequence_capture_limit})",
            flush=True,
        )

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    follow_warned_missing_target = False
    follow_prev_eye_w = None
    follow_prev_target_w = None
    follow_prev_pair_dir_w = None
    follow_prev_framing_distance = None

    def _update_follow_camera():
        nonlocal follow_warned_missing_target
        nonlocal follow_prev_eye_w, follow_prev_target_w
        nonlocal follow_prev_pair_dir_w, follow_prev_framing_distance
        if not args_cli.follow_camera:
            return
        try:
            if not hasattr(raw_env, "_robot"):
                return
            num_envs = int(getattr(raw_env, "num_envs", 1))
            env_id = min(max(int(args_cli.follow_env_id), 0), max(num_envs - 1, 0))
            root_pos_w = raw_env._robot.data.root_pos_w[env_id]
            root_quat_w = raw_env._robot.data.root_quat_w[env_id]
            follow_subject = str(getattr(args_cli, "follow_subject", "drone")).lower()

            drone_forward_w = math_utils.quat_apply(
                root_quat_w.unsqueeze(0),
                torch.tensor([[1.0, 0.0, 0.0]], device=root_pos_w.device),
            )[0]
            # Keep follow camera level: ignore roll/pitch by projecting forward onto the XY plane.
            drone_forward_w[2] = 0.0
            drone_forward_w = drone_forward_w / torch.clamp(torch.linalg.norm(drone_forward_w), min=1e-6)

            anchor_pos_w = root_pos_w
            forward_w = drone_forward_w

            if follow_subject == "target":
                if hasattr(raw_env, "_desired_pos_w"):
                    anchor_pos_w = raw_env._desired_pos_w[env_id]
                    forward_w = anchor_pos_w - root_pos_w
                    forward_w[2] = 0.0
                    if torch.linalg.norm(forward_w) > 1e-6:
                        forward_w = forward_w / torch.clamp(torch.linalg.norm(forward_w), min=1e-6)
                    else:
                        forward_w = drone_forward_w
                elif not follow_warned_missing_target:
                    print(
                        "[WARN] --follow_subject target requested but env has no _desired_pos_w. "
                        "Falling back to drone follow.",
                        flush=True,
                    )
                    follow_warned_missing_target = True

            if follow_subject == "both" and hasattr(raw_env, "_desired_pos_w"):
                target_pos_w = raw_env._desired_pos_w[env_id]
                mid_w = 0.5 * (root_pos_w + target_pos_w)
                pair_vec_w = target_pos_w - root_pos_w
                pair_vec_w[2] = 0.0
                pair_dist = torch.linalg.norm(target_pos_w - root_pos_w)
                if torch.linalg.norm(pair_vec_w) > 1e-6:
                    candidate_pair_dir_w = pair_vec_w / torch.clamp(torch.linalg.norm(pair_vec_w), min=1e-6)
                else:
                    candidate_pair_dir_w = drone_forward_w

                lock_distance = max(0.0, float(args_cli.follow_both_lock_distance))
                if (
                    follow_prev_pair_dir_w is not None
                    and float(pair_dist.item()) <= lock_distance
                ):
                    pair_dir_w = follow_prev_pair_dir_w
                else:
                    pair_dir_w = candidate_pair_dir_w
                    follow_prev_pair_dir_w = pair_dir_w.clone()

                framing_distance_raw = max(float(args_cli.follow_distance), float(pair_dist.item()) * 1.25 + 1.0)
                if follow_prev_framing_distance is None:
                    follow_prev_framing_distance = framing_distance_raw
                zoom_alpha = max(0.0, min(float(args_cli.follow_smooth_alpha), 1.0))
                framing_distance = (1.0 - zoom_alpha) * float(follow_prev_framing_distance) + zoom_alpha * framing_distance_raw
                follow_prev_framing_distance = framing_distance
                eye_w = mid_w - pair_dir_w * framing_distance
                eye_w[2] = mid_w[2] + float(args_cli.follow_height)
                target_w = mid_w.clone()
            else:
                eye_w = anchor_pos_w - forward_w * float(args_cli.follow_distance)
                eye_w[2] = eye_w[2] + float(args_cli.follow_height)
                if follow_subject == "target" and hasattr(raw_env, "_desired_pos_w"):
                    target_w = anchor_pos_w.clone()
                else:
                    target_w = anchor_pos_w + forward_w * float(args_cli.follow_look_ahead)
                    target_w[2] = anchor_pos_w[2]

            smooth_alpha = max(0.0, min(float(args_cli.follow_smooth_alpha), 1.0))
            if smooth_alpha > 0.0 and follow_prev_eye_w is not None and follow_prev_target_w is not None:
                eye_w = follow_prev_eye_w + (eye_w - follow_prev_eye_w) * smooth_alpha
                target_w = follow_prev_target_w + (target_w - follow_prev_target_w) * smooth_alpha
            follow_prev_eye_w = eye_w.clone()
            follow_prev_target_w = target_w.clone()

            raw_env.sim.set_camera_view(eye=eye_w.tolist(), target=target_w.tolist())
        except Exception:
            # In headless/non-viewport mode this can fail; ignore to keep playback running.
            return

    if args_cli.follow_camera:
        print(
            "[INFO] Follow camera enabled: "
            f"subject={args_cli.follow_subject}, "
            f"env_id={args_cli.follow_env_id}, dist={args_cli.follow_distance}, "
            f"height={args_cli.follow_height}, look_ahead={args_cli.follow_look_ahead}",
            flush=True,
        )
        _update_follow_camera()
    if args_cli.test_fixed_distance is not None:
        print(f"[INFO] Test fixed distance: {float(args_cli.test_fixed_distance):.3f} m", flush=True)
    if args_cli.test_fixed_speed is not None:
        print(f"[INFO] Test fixed speed: {float(args_cli.test_fixed_speed):.3f} m/s", flush=True)
    if args_cli.test_num_episodes is not None:
        print(f"[INFO] Test episode limit: {int(args_cli.test_num_episodes)}", flush=True)

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            try:
                obs, _, _, _, _ = env.step(actions)
            except Exception as exc:
                err_msg = str(exc).lower()
                if "invalidated" in err_msg or "failed to get" in err_msg or "from backend" in err_msg:
                    print(
                        "[INFO] Simulation view became invalid during playback; "
                        "stopping play loop cleanly.",
                        flush=True,
                    )
                    break
                raise
            _update_follow_camera()
            if args_cli.image_sequence:
                rep.orchestrator.step(
                    delta_time=0.0,
                    pause_timeline=False,
                    rt_subframes=max(0, int(args_cli.image_sequence_rt_subframes)),
                )
                image_sequence_frame_count += 1
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.image_sequence and image_sequence_frame_count >= image_sequence_capture_limit:
            print(f"[INFO] Reached image-sequence frame limit: {image_sequence_frame_count}", flush=True)
            break
        if args_cli.test_num_episodes is not None:
            total_episodes = getattr(raw_env, "_test_total_episodes", None)
            if total_episodes is not None and int(total_episodes) >= int(args_cli.test_num_episodes):
                print(f"[INFO] Reached test episode limit: {int(total_episodes)}", flush=True)
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if image_sequence_writer is not None:
        rep.orchestrator.wait_until_complete()
        image_sequence_writer.detach()
        image_sequence_render_product.destroy()
        print(f"[INFO] Image-sequence frames saved to: {image_sequence_output_dir}", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
