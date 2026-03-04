from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler


def _scaler_summary(scaler: Any) -> dict:
    """Summarize scaler internals without assuming a specific skrl version layout."""
    if scaler is None:
        return {"exists": False}

    summary: dict[str, Any] = {
        "exists": True,
        "type": type(scaler).__name__,
        "device": str(getattr(scaler, "device", "unknown")),
    }
    if hasattr(scaler, "state_dict"):
        try:
            state = scaler.state_dict()
            summary["state_keys"] = sorted(list(state.keys()))
            tensor_shapes = {}
            scalar_items = {}
            for k, v in state.items():
                if torch.is_tensor(v):
                    tensor_shapes[k] = list(v.shape)
                elif isinstance(v, (int, float, bool)):
                    scalar_items[k] = v
            if tensor_shapes:
                summary["tensor_shapes"] = tensor_shapes
            if scalar_items:
                summary["scalars"] = scalar_items
        except Exception as exc:
            summary["state_dict_error"] = str(exc)
    return summary


def _get_obs_dim(env) -> int:
    obs_space = getattr(env, "observation_space", None)
    if obs_space is None:
        raise RuntimeError("Unable to resolve env.observation_space for preprocessor reset")
    return int(gym.spaces.flatdim(obs_space))


def _set_preprocessor_attr(agent, target: str, value):
    """Set preprocessor attr with robust fallback across skrl versions/wrappers."""
    direct_attr = f"{target}_preprocessor"
    if hasattr(agent, direct_attr):
        setattr(agent, direct_attr, value)
        return direct_attr

    candidates = [name for name in dir(agent) if target in name and "preprocessor" in name]
    for name in candidates:
        try:
            setattr(agent, name, value)
            return name
        except Exception:
            continue

    all_preproc_attrs = [name for name in dir(agent) if "preprocessor" in name]
    raise RuntimeError(
        f"Unable to set {direct_attr}. Available preprocessor-like attrs: {all_preproc_attrs}"
    )


def _get_preprocessor_attr(agent, target: str):
    direct_attr = f"{target}_preprocessor"
    if hasattr(agent, direct_attr):
        return getattr(agent, direct_attr), direct_attr

    candidates = [name for name in dir(agent) if target in name and "preprocessor" in name]
    for name in candidates:
        try:
            return getattr(agent, name), name
        except Exception:
            continue

    all_preproc_attrs = [name for name in dir(agent) if "preprocessor" in name]
    raise RuntimeError(
        f"Unable to read {direct_attr}. Available preprocessor-like attrs: {all_preproc_attrs}"
    )


def reset_preprocessors(agent, env, device, reset_state: bool = True, reset_value: bool = True) -> dict:
    """Reset skrl preprocessors for fine-tuning.

    state scaler size = obs_dim
    value scaler size = 1
    """
    result: dict[str, Any] = {
        "reset_state": bool(reset_state),
        "reset_value": bool(reset_value),
    }

    if reset_state:
        old_state, old_attr = _get_preprocessor_attr(agent, "state")
        obs_dim = _get_obs_dim(env)
        new_state = RunningStandardScaler(size=obs_dim, device=device)
        set_attr = _set_preprocessor_attr(agent, "state", new_state)
        result["state"] = {
            "attr": set_attr,
            "old_attr": old_attr,
            "obs_dim": obs_dim,
            "old": _scaler_summary(old_state),
            "new": _scaler_summary(new_state),
        }

    if reset_value:
        old_value, old_attr = _get_preprocessor_attr(agent, "value")
        new_value = RunningStandardScaler(size=1, device=device)
        set_attr = _set_preprocessor_attr(agent, "value", new_value)
        result["value"] = {
            "attr": set_attr,
            "old_attr": old_attr,
            "size": 1,
            "old": _scaler_summary(old_value),
            "new": _scaler_summary(new_value),
        }

    return result


def set_optimizer_lr(agent, lr: float) -> None:
    """Set LR for single/list/dict optimizer containers."""
    opts = None
    if hasattr(agent, "optimizers"):
        opts = agent.optimizers
    elif hasattr(agent, "optimizer"):
        opts = agent.optimizer

    if opts is None:
        raise RuntimeError("Unable to find optimizer(s) on agent")

    if isinstance(opts, dict):
        iterable = list(opts.values())
    elif isinstance(opts, (list, tuple)):
        iterable = list(opts)
    else:
        iterable = [opts]

    for opt in iterable:
        if opt is None:
            continue
        for pg in getattr(opt, "param_groups", []):
            pg["lr"] = float(lr)

