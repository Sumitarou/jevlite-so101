"""M4a report and video overlay on a fake run folder (no ROS)."""

import json

import numpy as np
import pytest

from scripts.sim import gate_demo_report as rep
from scripts.sim import kinematics as kin
from scripts.sim.gate_loop import gate_metrics
from scripts.sim.gate_video import BAND_H, make_overlay
from tests.test_m4a_gate_loop import run

HAVE_MESHES = all(
    (kin.DESC / "meshes" / n).is_file()
    for n in json.loads((kin.DESC / "assets.lock").read_text())["meshes"]
)
needs_meshes = pytest.mark.skipif(not HAVE_MESHES, reason="SO-101 meshes not fetched")


def _fake_run(tmp_path):
    out = tmp_path / "run"
    for name, zone in (("base_1", lambda t: "none"),
                       ("center_1", lambda t: "center" if 1.0 <= t < 1.5 else "none")):  # fmt: skip
        _, _, _, ticks = run(zone, seconds=3.0)
        for k in ticks:
            k["intruder"] = [0.075, 0.15 if k["gate"]["truth"] == "center" else 0.55, 0.105]
        bag = out / "bags" / f"ep_{name}"
        bag.mkdir(parents=True)
        (bag / "ticks.jsonl").write_text("".join(json.dumps(k) + "\n" for k in ticks))
        (bag / "judge.jsonl").write_text(json.dumps(
            {"t_release": 0.0, "choice": "CONTINUE", "conf": 0.95}) + "\n")  # fmt: skip
        m = gate_metrics(ticks, lambda q: np.asarray(q)[:, :3] * 0.01, react_s=0.1)
        e = {"scenario": name, "desc": name, "bag": bag.name, "metrics": m,
             "keys": [] if name == "base_1" else [[1.0, "center"]],
             "result": {"success": True, "error_m": 0.01}, "intruder_meas_err_max_m": 0.0}
        with (out / "results.jsonl").open("a") as f:
            f.write(json.dumps(e) + "\n")
    return out


@needs_meshes
def test_report_writes_figures_and_table(tmp_path):
    run_dir = _fake_run(tmp_path)
    out = tmp_path / "report"
    assert rep.main([str(run_dir), "--out", str(out), "--lang", "en"]) == 0
    assert (out / "base_1.png").stat().st_size > 10_000 and (out / "center_1.png").is_file()
    rows = json.loads((out / "report.json").read_text())["rows"]
    c = next(r for r in rows if r["scenario"] == "center_1")
    assert c["resumes"] == 2 and c["stop_after_center_s"] == 0.0
    assert c["base_same_window_cm_s"] is not None
    md = (out / "summary.md").read_text()
    assert "| center_1 | ○ |" in md and "| base_1 | ○ |" in md


def test_overlay_draws_the_state_band():
    _, _, _, ticks = run(lambda t: "none", seconds=1.0)
    ov = make_overlay(ticks, loop_t0=10.0, scenario="x")
    img = np.zeros((480, 640, 3), np.uint8)
    ov(img, 9.0)  # before the loop: untouched
    assert img.sum() == 0
    ov(img, 10.1)  # STOP at the start: red band (BGR)
    band = img[480 - BAND_H :, 300:302].reshape(-1, 3).mean(0)
    assert band[2] > band[1] and img[: 480 - BAND_H].sum() == 0


@needs_meshes
def test_report_compare_marks_the_slow_checks(tmp_path):
    run_dir = _fake_run(tmp_path)
    out = tmp_path / "cmp"
    assert rep.main([str(run_dir), "--compare", str(run_dir), "--out", str(out),
                     "--lang", "en"]) == 0  # fmt: skip
    comp = json.loads((out / "report.json").read_text())["compare"]["rows"]
    assert {r["scenario"] for r in comp} == {"base_1", "center_1"}
    assert all(r["checks"]["success"] for r in comp)
    assert "SLOW" in (out / "compare.md").read_text()


def test_slow_stats_exit_and_lag():
    from scripts.sim.gate_loop import slow_stats

    t = np.arange(10) * 0.1
    state = ["CONTINUE"] * 3 + ["SLOW"] * 4 + ["CONTINUE"] * 3
    speed = np.array([3, 3, 3, 1, 1, 1, 1, 5, 2, 2], float)
    q_now = np.zeros((10, 6))
    q_tgt = np.zeros((10, 6))
    q_tgt[6, 2] = 0.1  # lag at the end of SLOW
    st = slow_stats(t, state, speed, q_tgt, q_now, [False] * 10)
    assert st["speed_mean"] == 1 and st["exit_speed_max"] == 5
    assert st["lag_end_max"] == pytest.approx(0.1) and st["exits"] == [pytest.approx(0.6)]
    assert slow_stats(t, ["CONTINUE"] * 10, speed, q_tgt, q_now, [False] * 10) == {"slow_s": 0.0}
