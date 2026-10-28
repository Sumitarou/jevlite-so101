import csv
from pathlib import Path

import numpy as np
import pytest

from gate.safety_gate import Reason, load_gate_config
from jevlite.api import Choice
from jevlite.config import load_config
from scripts import sim_gate_demo as demo


@pytest.fixture(scope="module")
def scenario():
    return load_config("sim_demo.yaml")


@pytest.fixture(scope="module")
def res(scenario):
    return demo.simulate(load_gate_config(), scenario)


@pytest.fixture(scope="module")
def res_off(scenario):
    return demo.simulate(demo.without_rate_limit(load_gate_config()), scenario)


def _states(res, t0, t1):
    return {s for t, s in zip(res.t, res.state, strict=True) if t0 <= t < t1}


def _times(res, pred):
    return np.array([t for k, t in enumerate(res.t) if pred(k)])


@pytest.fixture(scope="module")
def obstacle(scenario):
    (ob,) = scenario["obstacles"]  # the demo has exactly one "blocked by an obstacle" window
    return float(ob["start"]), float(ob["end"])


def test_timeline_follows_the_story(res, obstacle):
    """Startup hold -> run -> obstacle (G9 STOP, retry) -> hand: STOP through flicker/stall/skew
    -> SLOW -> CONTINUE."""
    start, end = obstacle
    assert res.resume_times[0] == pytest.approx(0.5)
    assert _states(res, 0.0, 0.5) == {Choice.STOP}  # startup hold (G6)
    assert _states(res, 0.5, start) == {Choice.CONTINUE}
    assert Choice.STOP in _states(res, start, end)  # G9 stops the blocked arm
    assert _states(res, 3.0, 7.1) == {Choice.STOP}  # hand, flicker, stall, clock skew
    assert res.resume_times[-1] == pytest.approx(7.1)
    assert _states(res, 7.1, 7.9) == {Choice.SLOW}  # withdrawing, after hold
    assert _states(res, 7.9, 10.0) == {Choice.CONTINUE}


def test_g9_fires_only_because_of_the_obstacle(res, obstacle):
    start, end = obstacle
    fe = _times(res, lambda k: res.reason[k] is Reason.FOLLOW_ERROR)
    assert fe.size > 0
    assert fe.min() >= start and fe.max() < end + 0.1  # nowhere else (normal tracking is fine)
    first = fe.min()
    assert first - start < 0.5  # caught quickly after the arm got stuck


def test_g9_stop_resumes_via_g6_after_the_obstacle(res, obstacle):
    start, end = obstacle
    resumes_after_block = [t for t in res.resume_times if start < t < 3.0]
    assert resumes_after_block  # at least one retry / recovery before the hand arrives
    assert _states(res, end + 0.5, 3.0) == {Choice.CONTINUE}  # recovered after unblocking


def test_scenario_exercises_each_reason(res):
    reasons = set(res.reason)
    assert {Reason.OK, Reason.LOW_CONF, Reason.STALE, Reason.FUTURE, Reason.FOLLOW_ERROR} <= reasons


def test_arm_does_not_move_while_blocked(res, obstacle):
    start, end = obstacle
    idx = [k for k, t in enumerate(res.t) if start <= t < end]
    assert np.abs(res.q_now[idx] - res.q_now[idx[0]]).max() < 1e-9
    assert all(res.blocked[k] for k in idx)


def test_q_cmd_matches_g3_g4_g5_without_g8(res_off):
    res = res_off
    alpha = load_gate_config().slow_alpha
    for k, s in enumerate(res.state):
        q_now, q_target = res.q_now[k], res.q_target[k]
        if s is Choice.STOP:
            np.testing.assert_array_equal(res.q_cmd[k], q_now)
        elif s is Choice.SLOW:
            np.testing.assert_allclose(res.q_cmd[k], q_now + alpha * (q_target - q_now))
        else:
            np.testing.assert_array_equal(res.q_cmd[k], q_target)


def test_g8_bounds_every_command_change(res, scenario):
    """G8 (a): the command itself changes at most max_speed * dt per step (STOP excepted)."""
    limit = np.array(load_gate_config().max_speed) / scenario["control_hz"]
    for k in range(1, len(res.t)):
        if res.state[k] is not Choice.STOP:
            assert np.all(np.abs(res.q_cmd[k] - res.q_cmd[k - 1]) <= limit + 1e-9), k


def test_g8_removes_the_resume_jump(res, res_off, scenario):
    """The figure's point: without G8 the command jumps on resume; with G8 it does not."""
    limit = max(load_gate_config().max_speed) / scenario["control_hz"]
    assert demo.max_jump(res_off) > 10 * limit
    assert demo.max_jump(res) <= limit + 1e-9
    assert any(res.rate_limited) and not any(res_off.rate_limited)


def test_g8_does_not_cripple_normal_tracking(res, res_off):
    """Regression for the q_now-referenced G8 (mean error was 11.5 there): steady CONTINUE
    tracking must stay close to no-G8. The window starts 0.6 s after the last resume (7.9 s)
    so the deliberate, rate-limited catch-up after a resume is not counted."""

    def err(r):
        run = (r.t >= 8.5) & (r.t < 10.0)
        return np.abs(r.q_target - r.q_now)[run].mean()

    assert err(res) < 3.0
    assert err(res) < 2 * err(res_off) + 1.0


def test_arm_does_not_move_while_stopped(res):
    idx = [k for k, t in enumerate(res.t) if 3.1 <= t < 7.1]
    drift = np.abs(res.q_now[idx] - res.q_now[idx[0]]).max()
    assert drift < 1e-6


def test_control_rate_must_be_multiple_of_judge_rate(scenario):
    with pytest.raises(ValueError):
        demo.simulate(load_gate_config(), {**scenario, "control_hz": 45})


def test_main_writes_png_and_csv(tmp_path: Path):
    out = tmp_path / "fig" / "demo.png"
    assert demo.main(["--out", str(out), "--lang", "en"]) == 0
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with out.with_suffix(".csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 500  # 10 s at 50 Hz
    assert sum(int(r["on_resume"]) for r in rows) >= 3  # start, after obstacle, after hand
    cols = {"t", "state", "requested", "reason", "rate_limited", "blocked", "shoulder_pan_cmd"}
    assert cols <= set(rows[0])
    assert any(r["reason"] == "follow_error" for r in rows)


def test_main_no_rate_limit_and_no_csv(tmp_path: Path):
    out = tmp_path / "off.png"
    assert demo.main(["--out", str(out), "--lang", "en", "--no-rate-limit", "--no-csv"]) == 0
    assert out.exists()
    assert not out.with_suffix(".csv").exists()


def test_segment_labels_do_not_overlap():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    xs = [1.0, 6.2, 6.5, 7.0, 9.0]  # three labels crowded together, two far apart
    texts = [ax.text(x, 0, "a long segment label", ha="center", va="bottom") for x in xs]
    rows = demo.place_without_overlap(texts, fig)
    assert rows[0] == 0 and rows[-1] == 0
    assert len({rows[1], rows[2], rows[3]}) == 3
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (i, j)
    plt.close(fig)


# --- report metrics (scripts/sim_gate_report.py) ---------------------------------------


@pytest.fixture(scope="module")
def report_rows(scenario):
    from scripts import sim_gate_report as report

    return {r["variant"]: r for r in report.run_variants(load_gate_config(), scenario)}


def test_metrics_compare_the_four_settings(report_rows, scenario):
    limit = max(load_gate_config().max_speed) / scenario["control_hz"]
    r1, r2, r3, r4 = (report_rows[k] for k in "1234")
    # G8: the resume jump disappears
    assert r1["max_cmd_step"] > 10 * limit and r4["max_cmd_step"] > 10 * limit
    assert r2["max_cmd_step"] <= limit + 1e-9 and r3["max_cmd_step"] <= limit + 1e-9
    # steady tracking is the same with and without G8
    for r in (r1, r2, r3, r4):
        assert r["steady_track_err"] < 3.0
    # G9: only the settings with G9 detect the obstacle, within 0.5 s
    assert not r1["obstacle_detected"] and r1["detect_time_s"] is None
    assert not r2["obstacle_detected"]
    assert r3["obstacle_detected"] and 0 < r3["detect_time_s"] < 0.5
    assert r4["obstacle_detected"]
    # G8 off + G9 on re-stops right after resuming; with G8 it does not
    assert r3["unnecessary_stops"] == 0
    assert r4["unnecessary_stops"] >= 2
    assert r4["continue_ratio"] < r3["continue_ratio"]


def test_judge_stop_right_after_resume_is_not_unnecessary(report_rows):
    """In (3) the hand at 3.0 s comes 0.64 s after the 2.36 s resume: counted as a re-stop,
    but not as unnecessary (the judge asked for it with a valid decision)."""
    r3 = report_rows["3"]
    assert r3["restops_within_window"] == 1
    assert r3["unnecessary_stops"] == 0


def test_report_writes_csv_and_md(tmp_path: Path):
    from scripts import sim_gate_report as report

    assert report.main(["--out", str(tmp_path), "--no-figures"]) == 0
    with (tmp_path / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["variant"] for r in rows] == ["1", "2", "3", "4"]
    assert list(rows[0]) == report.COLUMNS
    assert rows[0]["detect_time_s"] == "" and rows[2]["detect_time_s"] != ""
    md = (tmp_path / "summary.md").read_text()
    assert md.count("\n| [") == 4  # four table rows
    for col in ("指令の1ステップあたりの最大変化", "定常区間の平均追従誤差", "障害物の検出",
                "検出までの時間", "不要なSTOP", "CONTINUE の時間割合"):  # fmt: skip
        assert col in md
    assert "腕は動いていない" in md  # note on the 1.82 s step in figure (3)
    assert not list(tmp_path.glob("*.png"))


def test_follow_err_sweep_shows_the_trade_off(scenario):
    from scripts import sim_gate_report as report

    low, mid, high = report.follow_err_sweep(load_gate_config(), scenario, [1.0, 15.0, 40.0])
    assert low["false_follow_error_cycles"] > 0  # too tight: stops during normal tracking
    assert mid["false_follow_error_cycles"] == 0 and mid["detect_time_s"] is not None
    assert high["detect_time_s"] is None  # too loose: the 0.8 s obstacle is never detected


# --- figure layout ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def fig(res, scenario):
    import matplotlib.pyplot as plt

    f = demo.make_figure(res, scenario, lang="en")
    yield f
    plt.close(f)


def test_legends_are_outside_the_plots(fig):
    r = fig.canvas.get_renderer()
    for ax in fig.axes[:2]:
        leg = ax.get_legend()
        assert leg is not None
        assert leg.get_window_extent(r).x0 >= ax.get_window_extent(r).x1


def test_background_colours_have_a_legend(fig):
    labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    for c in Choice:
        assert f"background: {c.name}" in labels
    assert "blocked" in labels


def test_q_target_and_q_now_lines_are_distinguishable(fig):
    lines = {ln.get_label(): ln for ln in fig.axes[1].get_lines()}
    target = lines["shoulder_pan q_target"]
    now = lines["shoulder_pan q_now"]
    cmd = lines["shoulder_pan q_cmd"]
    assert target.get_linestyle() == "--" and now.get_linestyle() == "-"
    assert now.get_linewidth() >= 2 * target.get_linewidth()
    assert now.get_alpha() < target.get_alpha() <= cmd.get_alpha()
