"""M4b: the judge comparison (scripts/eval_sim_judges.py) with fake decisions."""

import numpy as np
import pytest

from jevlite.api import Choice, Decision
from scripts.eval_sim_judges import judge_report, run_judge, zone_of_choice
from scripts.sim.intruder import load_m4a
from scripts.sim.judge_data import subset_masks

INV = zone_of_choice(load_m4a()["zones"])


def _d(choice: Choice, conf: float, lat: float = 1.0) -> Decision:
    rest = (1 - conf) / 2
    probs = {c.name: (conf if c is choice else rest) for c in Choice}
    return Decision(choice, probs, 0.0, lat, "fake")


def test_zone_of_choice():
    assert INV == {"CONTINUE": "none", "SLOW": "edge", "STOP": "center"}
    with pytest.raises(ValueError):
        zone_of_choice({"none": "CONTINUE", "edge": "STOP", "center": "STOP"})


def test_judge_report_counts_and_confidence():
    rows = [{"zone": z, "d_boundary": d, "arm_near": a, "kind": "table"}
            for z, d, a in [("center", "0.005", "1"), ("center", "0.05", "0"),
                            ("edge", "0.05", "0"), ("none", "0.05", "0")]]  # fmt: skip
    decs = [_d(Choice.CONTINUE, 0.5), _d(Choice.STOP, 0.9), _d(Choice.STOP, 0.7),
            _d(Choice.CONTINUE, 0.95)]  # fmt: skip
    rep = judge_report(decs, rows, INV, subset_masks(rows, 0.01), 0.6, [1.0, 2.0, 3.0])
    a = rep["all"]
    assert a["accuracy"] == 0.5 and a["center_to_none"] == 1
    assert rep["near"]["n"] == 1 and rep["near"]["center_to_none"] == 1
    assert rep["arm_near"]["accuracy"] == 0.0
    c = rep["confidence"]
    assert c["unsafe"] == 1 and c["unsafe_below_min_conf"] == 1
    assert c["wrong_below_min_conf"] == 1  # the center -> CONTINUE at 0.5; edge -> STOP at 0.7
    assert sum(c["hist"]["right"]) == 2 and sum(c["hist"]["wrong"]) == 2
    assert rep["latency_ms"]["max"] == 3.0
    assert rep["_pred"].tolist() == [0, 2, 2, 0]


class _Counting:
    def __init__(self):
        self.calls = 0

    def judge(self, image, t_capture=None):
        self.calls += 1
        return _d(Choice.SLOW, 0.8, lat=float(self.calls))


def test_run_judge_warms_up_and_collects_bench_n():
    j = _Counting()
    imgs = [np.zeros((2, 2, 3), np.uint8)] * 3
    r = run_judge(j, [{}] * 3, imgs, warmup=2, bench_n=10)
    assert len(r["decisions"]) == 3 and len(r["latency_ms"]) == 10
    assert j.calls == 12 and r["latency_ms"][0] == 3.0  # warm-up calls not counted
