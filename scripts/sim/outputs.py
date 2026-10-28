"""A new output folder for each analysis run, so earlier results are never overwritten."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "sim"  # git-ignored


def new_run_dir(name: str, base: Path = BASE, now: datetime | None = None) -> Path:
    """Create and return base/<YYYYmmdd_HHMMSS>_<name>/ (with _2, _3, ... if it exists)."""
    stem = f"{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}_{name}"
    path, i = base / stem, 1
    while path.exists():
        i += 1
        path = base / f"{stem}_{i}"
    path.mkdir(parents=True)
    return path


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout

    return {
        "commit": run("rev-parse", "HEAD").strip(),
        "dirty": bool(run("status", "--porcelain").strip()),
    }
