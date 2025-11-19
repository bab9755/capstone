from functools import lru_cache
from typing import Dict, Any

from helpers import load_config
from constants import GROUND_TRUTH_LIBRARY


@lru_cache(maxsize=1)
def _get_config() -> Dict[str, Any]:
    return load_config() or {}


@lru_cache(maxsize=1)
def get_context_settings() -> Dict[str, Any]:
    config = _get_config()
    return config.get("context", {})


@lru_cache(maxsize=1)
def get_metric() -> str:
    config = _get_config()
    return config.get("metric", "cosine-bm25")


@lru_cache(maxsize=1)
def get_active_profile_key() -> str:
    config = _get_config()
    experiments = config.get("experiments", {})
    active_key = experiments.get("active_profile")
    profiles = experiments.get("profiles", {})
    if not active_key:
        raise ValueError("No active_profile set under experiments in configs.yaml")
    if active_key not in profiles:
        raise ValueError(f"Active profile '{active_key}' not defined under experiments.profiles")
    return active_key


@lru_cache(maxsize=1)
def get_active_profile() -> Dict[str, Any]:
    config = _get_config()
    experiments = config.get("experiments", {})
    profiles = experiments.get("profiles", {})
    active_key = get_active_profile_key()
    return profiles.get(active_key, {})


@lru_cache(maxsize=1)
def get_ground_truth_bundle() -> Dict[str, Any]:
    profile = get_active_profile()
    key = profile.get("ground_truth_key")
    if not key:
        raise ValueError("ground_truth_key must be defined in the active profile")
    if key not in GROUND_TRUTH_LIBRARY:
        raise ValueError(f"Unknown ground truth key '{key}'. Available keys: {list(GROUND_TRUTH_LIBRARY.keys())}")
    return GROUND_TRUTH_LIBRARY[key]


@lru_cache(maxsize=1)
def get_runtime_settings() -> Dict[str, Any]:
    config = _get_config()
    profile = get_active_profile()
    overrides = config.get("swarm_overrides") or {}

    ground_truth_bundle = get_ground_truth_bundle()
    num_subject_agents = profile.get("num_subject_agents") or len(ground_truth_bundle.get("snippets", []))

    environment_cfg = profile.get("environment") or {}
    width = environment_cfg.get("width")
    height = environment_cfg.get("height")
    if width is None or height is None:
        raise ValueError(f"Environment dimensions must be set for profile '{get_active_profile_key()}'")

    swarm_type = overrides.get("swarm_type") or profile.get("swarm_type") or "self_learning"
    social_override = overrides.get("social_learning_enabled")
    if social_override is None:
        social_learning_enabled = swarm_type == "social_learning"
    else:
        social_learning_enabled = bool(social_override)

    visualization_cfg = config.get("visualization") or {}
    live_plot_cfg = visualization_cfg.get("live_plot") or {}
    live_plot_enabled = live_plot_cfg.get("enabled")
    if live_plot_enabled is None:
        live_plot_enabled = True
    else:
        live_plot_enabled = bool(live_plot_enabled)

    update_interval_raw = live_plot_cfg.get("update_interval_ms")
    if update_interval_raw is None:
        live_plot_interval_ms = 3000
    else:
        try:
            live_plot_interval_ms = int(update_interval_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("visualization.live_plot.update_interval_ms must be an integer") from exc
        if live_plot_interval_ms <= 0:
            raise ValueError("visualization.live_plot.update_interval_ms must be positive")

    return {
        "profile_key": get_active_profile_key(),
        "swarm_type": swarm_type,
        "social_learning_enabled": social_learning_enabled,
        "environment": {
            "width": width,
            "height": height,
        },
        "agents": {
            "knowledge": profile.get("num_knowledge_agents", 1),
            "subjects": num_subject_agents,
        },
        "context": get_context_settings(),
        "metric": get_metric(),
        "ground_truth": ground_truth_bundle,
        "visualization": {
            "live_plot": {
                "enabled": live_plot_enabled,
                "update_interval_ms": live_plot_interval_ms,
            }
        },
    }

