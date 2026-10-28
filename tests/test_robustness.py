"""scripts/sim/robustness_core.py: M3e conditions, block perturbations and tracking metrics."""

import json
import math

import numpy as np
import pytest

from scripts.sim import demo_plan as dp
from scripts.sim import kinematics as kin
from scripts.sim import robustness_core as rc

RCFG, CFG = rc.load_configs()
MCFG = RCFG["metrics"]
SIZE = CFG["block"]["size"]
HAVE_MESHES = all(
    (kin.DESC / "meshes" / n).is_file()
    for n in json.loads((kin.DESC / "assets.lock").read_text())["meshes"]
)
needs_meshes = pytest.mark.skipif(not HAVE_MESHES, reason="SO-101 meshes not fetched")


# ---------- conditions ----------


def test_start_suite_conditions():
    a = rc.build_conditions(RCFG, CFG, "start")
    assert [c.id for c in a] == [c.id for c in rc.build_conditions(RCFG, CFG, "start")]
    assert len(a) == 17 and len({c.id for c in a}) == 17
    assert {c.n_action_steps for c in a} == {100}
    by = {c.id: c for c in a}
    assert by["center"].start == pytest.approx((0.0, -0.04, 0.0))
    assert not any(by["center"].outside.values()) and not any(by["x+2.5"].outside.values())
    assert by["x+3.0"].outside == {"x": True, "y": False, "yaw": False}
    assert by["x-5.0"].start[0] == pytest.approx(-0.05)
    assert by["yaw+25"].outside["yaw"] and math.degrees(by["yaw-35"].start[2]) == pytest.approx(-35)
    assert by["y-0.12"].start[1] == pytest.approx(-0.12) and by["y-0.12"].outside["y"]
    assert by["y+0.04"].start[1] == pytest.approx(0.04) and by["y+0.04"].outside["y"]
    assert all(c.perturbation is None for c in a)


def test_perturb_suite_conditions_are_grouped_by_setting():
    a = rc.build_conditions(RCFG, CFG, "perturb")
    assert len(a) == 7 * 3 + 2 and len({c.id for c in a}) == len(a)
    ns = [c.n_action_steps for c in a]
    assert ns == sorted(ns, key=[100, 30, 10].index)  # one server start per setting
    ids = {c.id for c in a}
    assert {"baseline_n30", "baseline_n10", "push_x+2_n100", "pre_x+2_n10"} <= ids
    assert "baseline_n100" not in ids  # the start suite's centre stands in
    p = next(c for c in a if c.id == "push_yaw-15_n30").perturbation
    assert p == {"name": "push_yaw-15", "t_s": 4.0, "dx": 0.0, "dyaw_deg": -15.0, "phase": "push"}
    assert all(c.start == pytest.approx((0.0, -0.04, 0.0)) for c in a)


def test_repeats_add_suffixed_ids():
    a = rc.build_conditions({**RCFG, "repeats": 2}, CFG, "start")
    assert len(a) == 34 and "center_r2" in {c.id for c in a}


def test_unknown_suite():
    with pytest.raises(ValueError):
        rc.build_conditions(RCFG, CFG, "nope")


def test_video_ids_exist():
    for suite, ids in RCFG["videos"].items():
        assert set(ids) <= {c.id for c in rc.build_conditions(RCFG, CFG, suite)}


# ---------- perturbation ----------


def test_x_shift_keeps_y_and_yaw():
    new, dy = rc.perturbed_pose((0.01, -0.02, 0.1), 0.02, 0.0, SIZE, 0.002)
    assert new == pytest.approx((0.03, -0.02, 0.1)) and dy == 0.0


@pytest.mark.parametrize("yaw_deg", [-20, -5, 0, 5, 20])
@pytest.mark.parametrize("dyaw_deg", [-15, 15])
def test_turned_block_clears_the_jaws(yaw_deg, dyaw_deg):
    cur = (0.0, -0.03, math.radians(yaw_deg))
    jaw_y = rc.block_corners(cur, SIZE)[:, 1].min()  # the jaws touch the block's -y side
    new, dy = rc.perturbed_pose(cur, 0.0, math.radians(dyaw_deg), SIZE, 0.002)
    assert new[2] == pytest.approx(cur[2] + math.radians(dyaw_deg)) and dy >= 0.002
    assert rc.block_corners(new, SIZE)[:, 1].min() >= jaw_y + 0.002 - 1e-9


def test_yaw_of_quaternion():
    a = 0.3
    assert rc.yaw_of(0, 0, math.sin(a / 2), math.cos(a / 2)) == pytest.approx(a)


# ---------- metrics (fake FK: the state is the TCP itself) ----------


def fake_fk(q):
    return np.atleast_2d(np.asarray(q, float))[:, :3]


def _ticks(tcp_x, block_x, event_at=None, infer_every=100, rate=30):
    out = []
    for i, (tx, bx) in enumerate(zip(tcp_x, block_x, strict=True)):
        k = {"i": i, "t": i / rate, "inferred": i % infer_every == 0,
             "state": [tx, -0.075, 0.005, 0.0, 0.0], "block": [bx, -0.04, 0.0]}  # fmt: skip
        if i == event_at:
            k["event"] = {"name": "x+2", "from": [0.0, -0.04, 0.0], "to": [0.02, -0.04, 0.0],
                          "set_pose_ok": True}  # fmt: skip
        out.append(k)
    return out


def test_reaim_time_when_the_tcp_follows():
    n, ev = 120, 40
    block = [0.0] * ev + [0.02] * (n - ev)
    tcp = np.clip((np.arange(n) - 50) / 21, 0, 1) * 0.02  # moves over ticks 50..71
    m = rc.episode_metrics(_ticks(tcp, block, ev), fake_fk, CFG, MCFG)
    p = m["perturb"]
    assert p["perturb_ok"] and p["t_event"] == pytest.approx(40 / 30, abs=1e-3)
    assert p["first_inference_after_s"] == pytest.approx(60 / 30, abs=1e-3)  # tick 100
    assert p["gap_after"] == pytest.approx(0.02)
    assert p["reaim_s"] == pytest.approx((66 - 40) / 30, abs=1e-3)  # |gap| <= 5 mm from tick 66
    assert p["gap_end"] == pytest.approx(0.0, abs=1e-6)


def test_no_reaim_when_the_tcp_stays():
    n, ev = 120, 40
    m = rc.episode_metrics(_ticks([0.0] * n, [0.0] * ev + [0.02] * (n - ev), ev), fake_fk, CFG,
                           MCFG)  # fmt: skip
    assert m["perturb"]["reaim_s"] is None and m["perturb"]["gap_end"] == pytest.approx(0.02)


def test_the_set_pose_jump_is_not_counted_as_touched():
    n, ev = 60, 10
    ticks = _ticks([0.0] * n, [0.0] * ev + [0.02] * (n - ev), ev)
    for k in ticks:
        k["state"][2] = 0.05  # the hand stays high (as with the hold policy)
    m = rc.episode_metrics(ticks, fake_fk, CFG, MCFG)
    assert m["block_moved_m"] == 0.0 and not m["touched"] and m["perturb"]["perturb_ok"]
    for k in ticks[30:]:
        k["block"][1] = -0.03  # pushed 1 cm after the move
    m = rc.episode_metrics(ticks, fake_fk, CFG, MCFG)
    assert m["block_moved_m"] == pytest.approx(0.01) and m["touched"]


def test_failed_move_is_flagged():
    n, ev = 60, 10
    m = rc.episode_metrics(_ticks([0.0] * n, [0.0] * n, ev), fake_fk, CFG, MCFG)
    assert m["perturb"]["perturb_ok"] is False and m["perturb"]["perturb_err_x"] == 0.02


def test_window_needs_the_tcp_low_and_just_behind_the_block():
    ticks = _ticks([0.0] * 10, [0.0] * 10)
    for k in ticks[:5]:
        k["state"][2] = 0.05  # high
    assert rc.episode_metrics(ticks, fake_fk, CFG, MCFG)["push_ticks"] == 5
    for k in ticks:
        k["state"][1] = -0.2  # far behind
    m = rc.episode_metrics(ticks, fake_fk, CFG, MCFG)
    assert m["push_ticks"] == 0 and m["tcp_x_low"] is None and "perturb" not in m


def test_x_slope():
    rows = [{"name": n, "block_start": [x, -0.04, 0.0], "metrics": {"tcp_x_low": y}}
            for n, x, y in [("center", 0.0, 0.0), ("x+3.0", 0.03, 0.03), ("x-3.0", -0.03, -0.03),
                            ("yaw+25", 0.0, 0.05)]]  # fmt: skip
    assert rc.x_slope(rows)["slope"] == pytest.approx(1.0) and rc.x_slope(rows)["points"] == 3
    flat = [{**r, "metrics": {"tcp_x_low": 0.001}} for r in rows]
    assert rc.x_slope(flat)["slope"] == pytest.approx(0.0, abs=1e-9)
    assert rc.x_slope(rows[:1]) is None


def test_x_slopes_inside_and_outside_the_demo_range():
    """Follows inside the demo range (+-2.5 cm), stops at its edge outside (the M3e shape)."""
    xs = [-0.05, -0.04, -0.035, -0.03, -0.025, 0.0, 0.025, 0.03, 0.035, 0.04, 0.05]
    rows = [{"name": "x", "block_start": [x, -0.04, 0.0],
             "metrics": {"tcp_x_low": 0.9 * min(max(x, -0.025), 0.025)}} for x in xs]  # fmt: skip
    rows.append(
        {"name": "yaw+25", "block_start": [0.0, -0.04, 0.4], "metrics": {"tcp_x_low": 0.03}}
    )
    s = rc.x_slopes(rows, CFG["demo"])
    assert s["inside"]["slope"] == pytest.approx(0.9) and s["inside"]["points"] == 3  # edges in
    assert s["inside"]["x_range_m"] == [-0.025, 0.025]
    for k in ("outside_minus", "outside_plus"):
        assert s[k]["slope"] == pytest.approx(0.0, abs=1e-9) and s[k]["points"] == 4
    assert s["outside_plus"]["intercept_m"] == pytest.approx(0.0225)
    assert 0.3 < s["all"]["slope"] < 0.7 and s["all"]["points"] == 11  # one line hides the shape
    assert rc.x_slopes(rows[4:7], CFG["demo"])["outside_plus"] is None


# ---------- metrics with the real FK on planned demos ----------


@pytest.fixture(scope="module")
def planner():
    return dp.Planner(CFG)


@pytest.fixture(scope="module")
def fk(planner):
    return rc.TcpFK(CFG, planner.robot)


def demo_ticks(plan: dp.Plan, fk) -> list[dict]:
    """Ticks of a planned demo: the block sits still until the jaws reach it, then rides on them."""
    tcp = fk(plan.q[:, :5])
    x, y, yaw = plan.block_start
    reach = dp.half_extent(SIZE, yaw - math.pi / 2) + CFG["demo"]["jaw_half_thickness"]
    by, out = y, []
    for i, q in enumerate(plan.q):
        if tcp[i, 2] < 0.012 and abs(tcp[i, 0] - x) < 0.02:
            by = max(by, tcp[i, 1] + reach)
        out.append({"i": i, "t": i / 30, "inferred": i == 0, "state": list(q[:5]),
                    "block": [x, by, yaw]})  # fmt: skip
    return out


@needs_meshes
def test_planned_demos_follow_the_block(planner, fk):
    rows = []
    for seed in (100, 104, 106, 109):
        plan = planner.plan(seed)
        m = rc.episode_metrics(demo_ticks(plan, fk), fk, CFG, MCFG)
        assert m["push_ticks"] > 30
        assert m["gap_abs_max"] < 0.003  # the TCP pushes at the block's x
        assert m["touched"] and m["block_moved_m"] > 0.05
        rows.append({"name": "x", "block_start": list(plan.block_start), "metrics": m})
    assert rc.x_slope(rows)["slope"] == pytest.approx(1.0, abs=0.05)


@needs_meshes
def test_hand_at_home_touches_nothing(planner, fk):
    plan = planner.plan(100)
    ticks = demo_ticks(plan, fk)
    for k in ticks:
        k["state"] = list(plan.q[0, :5])
        k["block"] = list(plan.block_start)
    m = rc.episode_metrics(ticks, fk, CFG, MCFG)
    assert m["push_ticks"] == 0 and not m["touched"] and m["block_moved_m"] == 0.0


@needs_meshes
def test_first_command_matches_the_plan(planner):
    for seed in (100, 101, 106):
        plan = planner.plan(seed)
        np.testing.assert_allclose(planner.first_command(plan.block_start), plan.q[0], atol=1e-12)
    q = planner.first_command((0.05, -0.04, 0.0))  # outside the demo range: no checks, a pose
    assert q.shape == (6,) and np.isfinite(q).all()


@needs_meshes
def test_report_from_fake_runs(tmp_path, planner, fk):
    """robustness_report.py end to end on two fake run folders (planned demos as the ticks)."""
    from scripts.sim import robustness_report as rr

    def run_dir(suite, conds):
        d = tmp_path / suite
        (d / "bags").mkdir(parents=True)
        lines = []
        for k, c in enumerate(conds):
            plan = planner.plan(100 + k)
            ticks = demo_ticks(plan, fk)
            if c.perturbation:
                ticks[120]["event"] = {"name": c.name, "from": ticks[120]["block"],
                                       "to": ticks[120]["block"], "set_pose_ok": True}  # fmt: skip
            bag = d / "bags" / f"ep_{k:04d}_{c.id}"
            bag.mkdir()
            (bag / "ticks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ticks))
            lines.append({**c.meta(), "status": "done", "bag": bag.name,
                          "block_start": list(plan.block_start),
                          "result": {"success": k % 2 == 0, "error_m": 0.01,
                                     "yaw_deg": 1.0}})  # fmt: skip
        (d / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (d / "summary.json").write_text(json.dumps({"suite": suite, "config": RCFG}))
        return d

    start = run_dir("start", rc.build_conditions(RCFG, CFG, "start")[:4])
    pert = rc.build_conditions(RCFG, CFG, "perturb")
    pert = [c for c in pert if c.name in ("push_x+2", "baseline")]
    perturb = run_dir("perturb", pert)
    assert rr.main([str(start), str(perturb), "--out-base", str(tmp_path / "out")]) == 0
    out = next((tmp_path / "out").iterdir())
    for f in (
        "start.csv",
        "start.md",
        "perturb.csv",
        "perturb.md",
        "report.json",
        "fig1_start_x.png",
        "fig2_start_gap.png",
        "fig3_perturb.png",
        "fig4_trace.png",
    ):
        assert (out / f).is_file(), f
    rep = json.loads((out / "report.json").read_text())
    assert len(rep["start"]) == 4 and rep["x_slope"]["slope"] == pytest.approx(1.0, abs=0.05)
    assert rep["x_slopes"]["all"] == rep["x_slope"] and rep["x_slopes"]["inside"]["points"] >= 2
    names = {(r["perturbation"], r["n_action_steps"]) for r in rep["perturb"]}
    assert ("baseline", 100) in names and ("push_x+2", 10) in names
