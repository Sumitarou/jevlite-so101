"""train/export_intruder_frames.py: labels of the frames (no bag)."""

from pathlib import Path

import numpy as np
import yaml
from export_intruder_frames import label_rows, scenario_of

ROOT = Path(__file__).resolve().parents[2]
SIM = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
M4A = yaml.safe_load((ROOT / "configs" / "m4a.yaml").read_text())


def test_labels_follow_the_script_and_the_measured_pose():
    e = {"scenario": "center_1", "keys": [[1.0, "out"], [2.0, "center"]], "desc": ""}
    sc = scenario_of(e, M4A, SIM)
    t0 = 50.0
    stamps = t0 + np.arange(-0.5, 3.0, 1 / 30)
    # measured pose lags the script by one frame
    pt = stamps + 1 / 30
    pxy = np.array([sc.position(s - t0)[:2] for s in stamps])
    ticks = [{"t": round(k / 30, 4), "gate": {"state": "STOP" if k / 30 > 2 else "CONTINUE"}}
             for k in range(90)]  # fmt: skip
    rows = label_rows(sc, SIM, t0, t0 + 2.97, stamps, (pt, pxy), ticks, every=3)
    assert rows[0]["t_rel"] >= 0 and all(r["t_rel"] <= 2.97 + 1e-6 for r in rows)
    assert len(rows) == 30  # 90 frames inside the loop, every 3rd
    assert rows[0]["zone"] == "none" and rows[-1]["zone"] == "center"
    assert {r["zone"] for r in rows} == {"none", "edge", "center"}
    assert rows[-1]["gate_state"] == "STOP" and rows[0]["gate_state"] == "CONTINUE"
    agree = np.mean([r["zone"] == r["zone_meas"] for r in rows if r["zone_meas"]])
    assert agree > 0.8  # only frames at the zone changes can differ
