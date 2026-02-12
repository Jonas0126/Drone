from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


##
# Pre-defined configs
##
from .config.drone import DRONE_CFG  # isort: skip


class DroneTargetTouchEnvWindow(BaseEnvWindow):
    """Window manager for the target-touch environment."""

    def __init__(self, env: "DroneTargetTouchEnv", window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class DroneTargetTouchEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 30.0
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = DroneTargetTouchEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=8, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 2.5
    moment_scale = 0.05

    # reward scales
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = -0.002
    distance_to_goal_reward_scale = 26.0
    touch_bonus_reward = 70.0
    time_penalty_scale = 0.15
    distance_penalty_scale = 0.3
    death_penalty = 80.0
    far_away_termination_distance = 20.0

    # touch-task controls
    touch_radius = 0.48
    enable_touch_reward = True
    terminate_on_touch = True
    near_touch_outer_radius = 0.75
    near_touch_hover_speed_threshold = 0.12
    near_touch_vel_penalty_min_scale = 0.2

    # spawn ranges (relative to env origin)
    spawn_xy_min = -5.0
    spawn_xy_max = 5.0
    spawn_z_min = 1.0
    spawn_z_max = 5.0

    # hover curriculum (spawn range stages)
    curriculum_enabled = True
    curriculum_ramp_steps = 320_000
    curriculum_spawn_max_stages = (2.0, 4.0, 7.0)
