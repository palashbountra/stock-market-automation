"""
utils/config_loader.py
----------------------
Single source of truth for loading config.yaml.
All modules import `cfg` from here — never read yaml directly elsewhere.
"""

import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # project root


def load_config(path: str | None = None) -> dict:
    """Load and return config as a plain dict."""
    config_path = Path(path) if path else _ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Module-level singleton — import this everywhere
cfg: dict = load_config()
