from pathlib import Path
import yaml
from functools import lru_cache

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "configs.yaml"

@lru_cache(maxsize=1)
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config