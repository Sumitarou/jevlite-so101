"""M4b: which M4a scenarios to run with which judge, and their judge settings (numpy only).

configs/m4b.yaml lists the scenarios and two layers of overrides applied after the M4a judge
settings: judge_overrides for every judge (the oracle too, so it is the same comparison) and
image_judge_overrides for the image judges (latency from the judge itself, its own
probabilities).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from scripts.sim.intruder import Scenario

ROOT = Path(__file__).resolve().parents[2]
M4B_YAML = ROOT / "configs" / "m4b.yaml"
ORACLE = "oracle"


def load_m4b(path: Path = M4B_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


def suite_scenarios(scs: list[Scenario], m4b: dict, image_judge: bool) -> list[Scenario]:
    """The m4b scenarios (in the m4b.yaml order) with the overrides applied."""
    by_id = {s.id: s for s in scs}
    missing = [i for i in m4b["scenarios"] if i not in by_id]
    if missing:
        raise ValueError(f"m4b.yaml lists unknown scenarios {missing}")
    over = dict(m4b.get("judge_overrides") or {})
    if image_judge:
        over.update(m4b.get("image_judge_overrides") or {})
    return [replace(by_id[i], judge={**by_id[i].judge, **over}) for i in m4b["scenarios"]]


def judge_spec(m4b: dict, name: str, model: str | None = None) -> dict:
    """{kind, model} of an image judge; `model` (a folder) overrides the config's."""
    if name not in (m4b.get("judges") or {}):
        raise ValueError(f"unknown judge {name!r}; m4b.yaml has {sorted(m4b.get('judges') or {})}")
    spec = dict(m4b["judges"][name])
    if model is not None:
        spec["model"] = model
    if spec["kind"] in ("v1", "v2") and not spec.get("model"):
        raise ValueError(f"judge {name}: no model (set judges.{name}.model or --judge-model)")
    return spec
