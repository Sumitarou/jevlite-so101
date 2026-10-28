"""Load YAML files from configs/ (all tuning values live there, never in code)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    path = config_dir / name
    with path.open() as f:
        return yaml.safe_load(f) or {}
