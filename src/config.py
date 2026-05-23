"""Path helpers + dotenv loader robust to current working directory."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_YAML = CONFIG_DIR / "config.yaml"

_loaded = False


def load_env() -> None:
    """Load config/.env from project root. Idempotent — safe to call from any module."""
    global _loaded
    if _loaded:
        return
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
    _loaded = True


def load_config() -> dict:
    """Load config/config.yaml as dict."""
    import yaml
    with open(CONFIG_YAML) as f:
        return yaml.safe_load(f)


load_env()
