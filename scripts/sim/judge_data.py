"""M4b: the labelled rod images (scripts/sim/gen_intruder_data.py) and the judge metrics.

Shared by scripts/train_student.py and scripts/eval_sim_judges.py (uv env). Each data folder has
labels.csv and images/; rows carry the split (by episode), the zone and the subsets used in the
evaluation (near a boundary, arm near the rod).
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from jevlite.student import ZONES

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def load_rows(dirs: Sequence[Path]) -> list[dict]:
    """Rows of every labels.csv, with `path` (the image) and `data` (the folder name) added."""
    rows = []
    for d in dirs:
        d = Path(d)
        with (d / "labels.csv").open() as f:
            for r in csv.DictReader(f):
                if r["zone"] not in ZONES or r["split"] not in SPLITS:
                    raise ValueError(f"{d}: bad row {r['file']}: {r['zone']} / {r['split']}")
                rows.append({**r, "path": str(d / "images" / r["file"]), "data": d.name})
    if not rows:
        raise ValueError("no labelled images")
    return rows


def split_rows(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r["split"] == split]


def labels(rows: list[dict]) -> np.ndarray:
    return np.array([ZONES.index(r["zone"]) for r in rows], int)


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def rows_digest(rows: list[dict]) -> str:
    """Short hash of the image list and labels (for caches and the model's meta.json)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['data']}/{r['file']}:{r['zone']}:{r['split']};".encode())
    return h.hexdigest()[:16]


def subset_masks(rows: list[dict], near_m: float) -> dict[str, np.ndarray]:
    """all / near (footprint within near_m of a zone boundary) / arm_near / on_table (not park)."""
    d = np.array([float(r["d_boundary"]) for r in rows])
    return {
        "all": np.ones(len(rows), bool),
        "near": d <= near_m,
        "arm_near": np.array([r["arm_near"] == "1" for r in rows]),
        "on_table": np.array([r["kind"] != "park" for r in rows]),
    }


def zone_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, per-zone recall, confusion (rows = truth, columns = prediction, ZONES order) and
    the counts of the errors that matter for safety."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    k = len(ZONES)
    conf = np.zeros((k, k), int)
    np.add.at(conf, (y_true, y_pred), 1)
    n = int(conf.sum())
    rec = {z: (round(float(conf[i, i] / conf[i].sum()), 4) if conf[i].sum() else None)
           for i, z in enumerate(ZONES)}  # fmt: skip
    c, e, nn = ZONES.index("center"), ZONES.index("edge"), ZONES.index("none")
    return {
        "n": n,
        "accuracy": round(float(np.trace(conf) / n), 4) if n else None,
        "recall": rec,
        "confusion": conf.tolist(),
        "center_to_none": int(conf[c, nn]),   # center -> CONTINUE (the one that must be 0)
        "center_to_edge": int(conf[c, e]),    # center -> SLOW
        "edge_to_none": int(conf[e, nn]),     # edge -> CONTINUE
        "under": int(np.sum(y_pred < y_true)),  # milder than the truth (unsafe side)
        "over": int(np.sum(y_pred > y_true)),   # stricter than the truth (safe side)
    }  # fmt: skip


def better(a: dict, b: dict | None) -> bool:
    """Model selection on val: higher accuracy; on a tie, fewer center -> none."""
    if b is None:
        return True
    return (a["accuracy"], -a["center_to_none"]) > (b["accuracy"], -b["center_to_none"])
