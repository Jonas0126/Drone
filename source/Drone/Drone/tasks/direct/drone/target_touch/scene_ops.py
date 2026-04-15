from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz


def setup_scene(env) -> None:
    """Build the robot, terrain, lights, and scene caches."""
    env._robot = Articulation(env.cfg.robot)
    env.scene.articulations["robot"] = env._robot

    env.cfg.terrain.num_envs = env.scene.cfg.num_envs
    env.cfg.terrain.env_spacing = env.scene.cfg.env_spacing
    env._terrain = env.cfg.terrain.class_type(env.cfg.terrain)

    env.scene.clone_environments(copy_from_source=False)

    if env.device == "cpu":
        env.scene.filter_collisions(global_prim_paths=[env.cfg.terrain.prim_path])

    if bool(getattr(env.cfg, "use_default_distant_light", False)):
        distant_light_cfg = sim_utils.DistantLightCfg(
            intensity=float(getattr(env.cfg, "default_distant_light_intensity", 1000.0)),
            exposure=float(getattr(env.cfg, "default_distant_light_exposure", 0.0)),
            angle=float(getattr(env.cfg, "default_distant_light_angle_deg", 1.0)),
            color=tuple(getattr(env.cfg, "default_distant_light_color", (1.0, 1.0, 1.0))),
            normalize=bool(getattr(env.cfg, "default_distant_light_normalize", False)),
        )
        light_rot_deg = getattr(env.cfg, "default_distant_light_euler_deg", (45.0, 0.0, 90.0))
        light_rot_rad = torch.deg2rad(torch.tensor(light_rot_deg, dtype=torch.float))
        light_quat = quat_from_euler_xyz(
            light_rot_rad[0].unsqueeze(0), light_rot_rad[1].unsqueeze(0), light_rot_rad[2].unsqueeze(0)
        )[0]
        distant_light_cfg.func(
            getattr(env.cfg, "default_distant_light_prim_path", "/World/defaultLight"),
            distant_light_cfg,
            orientation=tuple(float(v) for v in light_quat.tolist()),
        )

    if bool(getattr(env.cfg, "use_default_dome_light", True)):
        light_cfg = sim_utils.DomeLightCfg(
            intensity=float(getattr(env.cfg, "default_dome_light_intensity", 2000.0)),
            exposure=float(getattr(env.cfg, "default_dome_light_exposure", 0.0)),
            color=tuple(getattr(env.cfg, "default_dome_light_color", (0.75, 0.75, 0.75))),
            texture_file=getattr(env.cfg, "default_dome_light_texture_file", None),
            texture_format=str(getattr(env.cfg, "default_dome_light_texture_format", "automatic")),
        )
        dome_rot_deg = getattr(env.cfg, "default_dome_light_euler_deg", (0.0, 0.0, 0.0))
        dome_rot_rad = torch.deg2rad(torch.tensor(dome_rot_deg, dtype=torch.float))
        dome_quat = quat_from_euler_xyz(
            dome_rot_rad[0].unsqueeze(0), dome_rot_rad[1].unsqueeze(0), dome_rot_rad[2].unsqueeze(0)
        )[0]
        light_cfg.func(
            getattr(env.cfg, "default_dome_light_prim_path", "/World/Light"),
            light_cfg,
            orientation=tuple(float(v) for v in dome_quat.tolist()),
        )

    cache_scene_anchor_data(env)
    cache_scene_obstacle_data(env)


def cache_scene_anchor_data(env) -> None:
    """Cache demo-scene anchor centers and clearance radii."""
    env._scene_anchor_centers_w = None
    env._scene_anchor_safe_radius = None

    if not bool(getattr(env.cfg, "scene_anchor_enabled", False)):
        return

    prim_path_template = getattr(env.cfg, "scene_anchor_prim_path", None)
    search_root_template = getattr(env.cfg, "scene_anchor_search_root_path", None)
    search_name = getattr(env.cfg, "scene_anchor_search_prim_name", None)
    fallback_xy_cfg = getattr(env.cfg, "scene_anchor_fallback_xy", None)
    if prim_path_template is None and not (search_root_template and search_name) and fallback_xy_cfg is None:
        return

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as exc:
        print(f"[WARN][Touch] scene anchor unavailable: {exc}", flush=True)
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    centers = torch.full((env.scene.cfg.num_envs, 3), float("nan"), dtype=torch.float, device=env.device)
    safe_radii = torch.full((env.scene.cfg.num_envs,), -1.0, dtype=torch.float, device=env.device)
    clearance_m = max(float(getattr(env.cfg, "scene_anchor_clearance_m", 0.0)), 0.0)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    for env_id in range(env.scene.cfg.num_envs):
        prim_path = None
        prim = None
        if prim_path_template:
            prim_path = str(prim_path_template).format(env_id=env_id)
            prim = stage.GetPrimAtPath(prim_path)
        if (not prim or not prim.IsValid()) and search_root_template and search_name:
            search_root_path = str(search_root_template).format(env_id=env_id)
            search_root = stage.GetPrimAtPath(search_root_path)
            prim = find_descendant_prim_by_name(search_root, str(search_name))

        if prim and prim.IsValid():
            world_box = bbox_cache.ComputeWorldBound(prim).GetBox()
            box_min = world_box.GetMin()
            box_max = world_box.GetMax()
            center_x = 0.5 * (float(box_min[0]) + float(box_max[0]))
            center_y = 0.5 * (float(box_min[1]) + float(box_max[1]))
            center_z = 0.5 * (float(box_min[2]) + float(box_max[2]))
            dx = max(float(box_max[0]) - float(box_min[0]), 0.0)
            dy = max(float(box_max[1]) - float(box_min[1]), 0.0)
            half_diag_xy = 0.5 * ((dx * dx + dy * dy) ** 0.5)
            centers[env_id] = torch.tensor((center_x, center_y, center_z), dtype=torch.float, device=env.device)
            safe_radii[env_id] = float(half_diag_xy + clearance_m)
            continue

        if fallback_xy_cfg is not None and len(fallback_xy_cfg) >= 2:
            fallback_x = float(fallback_xy_cfg[0])
            fallback_y = float(fallback_xy_cfg[1])
            centers[env_id] = torch.tensor((fallback_x, fallback_y, 0.0), dtype=torch.float, device=env.device)
            safe_radii[env_id] = float(clearance_m)
            print(
                f"[WARN][Touch] scene anchor fallback to fixed XY for env_{env_id}: "
                f"({fallback_x:.5f}, {fallback_y:.5f})",
                flush=True,
            )
            continue

        print(f"[WARN][Touch] scene anchor unresolved for env_{env_id}: {prim_path}", flush=True)

    if torch.any(safe_radii > 0.0):
        env._scene_anchor_centers_w = centers
        env._scene_anchor_safe_radius = safe_radii


def find_descendant_prim_by_name(root_prim, target_name: str):
    """Recursively search for a descendant prim by name."""
    if root_prim is None or not root_prim.IsValid():
        return None

    target_name = str(target_name).lower()
    stack = list(root_prim.GetChildren())
    while stack:
        prim = stack.pop()
        if prim.GetName().lower() == target_name:
            return prim
        stack.extend(list(prim.GetChildren()))
    return None


def cache_scene_obstacle_data(env) -> None:
    """Cache static obstacle boxes for demo spawn clearance / obstacle avoidance."""
    env._scene_obstacle_boxes_xy_min = [None] * env.scene.cfg.num_envs
    env._scene_obstacle_boxes_xy_max = [None] * env.scene.cfg.num_envs
    env._scene_obstacle_boxes_xy_center = [None] * env.scene.cfg.num_envs

    if not bool(getattr(env.cfg, "scene_obstacle_avoidance_enabled", False)) and not bool(
        getattr(env.cfg, "scene_obstacle_spawn_clearance_enabled", False)
    ):
        return

    root_path_templates = tuple(getattr(env.cfg, "scene_obstacle_search_root_paths", ()))
    if len(root_path_templates) == 0:
        return

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as exc:
        print(f"[WARN][Touch] scene obstacle cache unavailable: {exc}", flush=True)
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox_margin = max(float(getattr(env.cfg, "scene_obstacle_bbox_margin_m", 0.0)), 0.0)
    min_size_xy = max(float(getattr(env.cfg, "scene_obstacle_min_size_xy_m", 0.0)), 0.0)
    min_height = max(float(getattr(env.cfg, "scene_obstacle_min_height_m", 0.0)), 0.0)

    for env_id in range(env.scene.cfg.num_envs):
        boxes_xy_min: list[tuple[float, float]] = []
        boxes_xy_max: list[tuple[float, float]] = []
        boxes_xy_center: list[tuple[float, float]] = []

        for root_template in root_path_templates:
            root_path = str(root_template).format(env_id=env_id)
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim or not root_prim.IsValid():
                continue

            source_prims = list(root_prim.GetChildren())
            if len(source_prims) == 0:
                source_prims = [root_prim]

            for source_prim in source_prims:
                if source_prim is None or not source_prim.IsValid():
                    continue
                try:
                    world_box = bbox_cache.ComputeWorldBound(source_prim).GetBox()
                except Exception:
                    continue
                box_min = world_box.GetMin()
                box_max = world_box.GetMax()
                dx = max(float(box_max[0]) - float(box_min[0]), 0.0)
                dy = max(float(box_max[1]) - float(box_min[1]), 0.0)
                dz = max(float(box_max[2]) - float(box_min[2]), 0.0)
                if max(dx, dy) < min_size_xy or dz < min_height:
                    continue

                min_x = float(box_min[0]) - bbox_margin
                min_y = float(box_min[1]) - bbox_margin
                max_x = float(box_max[0]) + bbox_margin
                max_y = float(box_max[1]) + bbox_margin
                if not torch.isfinite(torch.tensor((min_x, min_y, max_x, max_y), dtype=torch.float)).all():
                    continue

                boxes_xy_min.append((min_x, min_y))
                boxes_xy_max.append((max_x, max_y))
                boxes_xy_center.append((0.5 * (min_x + max_x), 0.5 * (min_y + max_y)))

        if len(boxes_xy_min) == 0:
            print(f"[WARN][Touch] no scene obstacle boxes cached for env_{env_id}", flush=True)
            continue

        env._scene_obstacle_boxes_xy_min[env_id] = torch.tensor(boxes_xy_min, dtype=torch.float, device=env.device)
        env._scene_obstacle_boxes_xy_max[env_id] = torch.tensor(boxes_xy_max, dtype=torch.float, device=env.device)
        env._scene_obstacle_boxes_xy_center[env_id] = torch.tensor(
            boxes_xy_center, dtype=torch.float, device=env.device
        )
        print(
            f"[INFO][Touch] cached {len(boxes_xy_min)} scene obstacle boxes for env_{env_id}",
            flush=True,
        )


def sample_scene_anchor_ring_xy(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Sample XY positions on the configured anchor ring / annulus."""
    if env._scene_anchor_centers_w is None or env._scene_anchor_safe_radius is None:
        raise RuntimeError("scene anchor cache is not initialized")
    if not torch.all(env._scene_anchor_safe_radius[env_ids] > 0.0):
        raise RuntimeError("scene anchor cache is unresolved for some envs")

    theta = torch.empty((len(env_ids),), device=env.device).uniform_(0.0, 2.0 * torch.pi)
    radii = env._scene_anchor_safe_radius[env_ids]
    spawn_radius_min_cfg = getattr(env.cfg, "scene_anchor_spawn_radius_min_m", None)
    spawn_radius_max_cfg = getattr(env.cfg, "scene_anchor_spawn_radius_max_m", None)
    if spawn_radius_min_cfg is not None or spawn_radius_max_cfg is not None:
        spawn_radius_min = float(spawn_radius_min_cfg if spawn_radius_min_cfg is not None else 0.0)
        spawn_radius_max = float(spawn_radius_max_cfg if spawn_radius_max_cfg is not None else spawn_radius_min)
        if spawn_radius_min > spawn_radius_max:
            spawn_radius_min, spawn_radius_max = spawn_radius_max, spawn_radius_min
        spawn_radius_min = max(spawn_radius_min, 0.0)
        min_radii = torch.clamp(env._scene_anchor_safe_radius[env_ids], min=spawn_radius_min)
        max_radii = torch.full_like(min_radii, max(spawn_radius_max, 0.0))
        max_radii = torch.maximum(max_radii, min_radii)
        if torch.any(max_radii > min_radii):
            u = torch.empty((len(env_ids),), device=env.device).uniform_(0.0, 1.0)
            radii = torch.sqrt(u * (max_radii**2 - min_radii**2) + min_radii**2)
        else:
            radii = min_radii
    x = env._scene_anchor_centers_w[env_ids, 0] + radii * torch.cos(theta)
    y = env._scene_anchor_centers_w[env_ids, 1] + radii * torch.sin(theta)
    return torch.stack((x, y), dim=1)


def sample_scene_anchor_spawn_xy(env, env_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Sample spawn XY from either the configured rectangle or the anchor ring."""
    if bool(getattr(env.cfg, "scene_anchor_spawn_rect_enabled", False)):
        x_min_cfg = getattr(env.cfg, "scene_anchor_spawn_rect_x_min", None)
        x_max_cfg = getattr(env.cfg, "scene_anchor_spawn_rect_x_max", None)
        y_min_cfg = getattr(env.cfg, "scene_anchor_spawn_rect_y_min", None)
        y_max_cfg = getattr(env.cfg, "scene_anchor_spawn_rect_y_max", None)
        if None not in (x_min_cfg, x_max_cfg, y_min_cfg, y_max_cfg):
            x_min = float(x_min_cfg)
            x_max = float(x_max_cfg)
            y_min = float(y_min_cfg)
            y_max = float(y_max_cfg)
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            spawn_xy = torch.empty((len(env_ids), 2), device=env.device, dtype=dtype)
            spawn_xy[:, 0] = torch.empty((len(env_ids),), device=env.device, dtype=dtype).uniform_(x_min, x_max)
            spawn_xy[:, 1] = torch.empty((len(env_ids),), device=env.device, dtype=dtype).uniform_(y_min, y_max)
            return spawn_xy
    return sample_scene_anchor_ring_xy(env, env_ids).to(dtype=dtype)


def enforce_scene_anchor_clearance(
    env, points_w: torch.Tensor, env_ids: torch.Tensor, extra_clearance: float = 0.0
) -> torch.Tensor:
    """Push positions outside the demo-scene anchor safe radius."""
    if env._scene_anchor_centers_w is None or env._scene_anchor_safe_radius is None:
        return points_w

    centers_xy = env._scene_anchor_centers_w[env_ids, :2]
    safe_radius = env._scene_anchor_safe_radius[env_ids] + max(float(extra_clearance), 0.0)
    valid_mask = safe_radius > 0.0
    if not torch.any(valid_mask):
        return points_w
    adjusted_points = points_w.clone()
    delta_xy = adjusted_points[:, :2] - centers_xy
    dist_xy = torch.linalg.norm(delta_xy, dim=1)
    inside_mask = valid_mask & (dist_xy < safe_radius)
    if not torch.any(inside_mask):
        return adjusted_points

    safe_delta = delta_xy[inside_mask]
    safe_dist = dist_xy[inside_mask]
    degenerate = safe_dist < 1e-6
    if torch.any(degenerate):
        fallback_angles = torch.empty((int(degenerate.sum().item()),), device=env.device).uniform_(0.0, 2.0 * torch.pi)
        safe_delta[degenerate, 0] = torch.cos(fallback_angles)
        safe_delta[degenerate, 1] = torch.sin(fallback_angles)
        safe_dist = torch.linalg.norm(safe_delta, dim=1)
    safe_dir = safe_delta / torch.clamp(safe_dist.unsqueeze(-1), min=1e-6)
    adjusted_points[inside_mask, 0] = centers_xy[inside_mask, 0] + safe_dir[:, 0] * safe_radius[inside_mask]
    adjusted_points[inside_mask, 1] = centers_xy[inside_mask, 1] + safe_dir[:, 1] * safe_radius[inside_mask]
    return adjusted_points
