"""train/cmd_stats.py: step / follow statistics, phases and the proposed gate values (no bags)."""

from pathlib import Path

import numpy as np
import yaml
from cmd_stats import episode_rows, phases, propose, summarize, to_md

from gate.safety_gate import load_gate_config
from units import ARM6

ROOT = Path(__file__).resolve().parents[2]


def test_episode_rows_step_and_follow():
    t_a = np.array([0, 10, 20, 30])
    action = np.array([[0.0], [0.1], [0.3], [0.3]])
    t_s = np.array([0, 30])
    state = np.array([[0.0], [0.3]])  # q_now(t) = 0.01 t
    step, follow, chunk = episode_rows(t_a, action, t_s, state, [True, False, True, False])
    assert np.allclose(step[:, 0], [0.1, 0.2, 0.0])
    assert np.allclose(follow[:, 0], [0.1, 0.1, 0.0])  # prev command vs q_now at the next tick
    assert chunk.tolist() == [False, True, False]


def test_phases_approach_descend_push_after():
    z = [0.05, 0.03, 0.02, 0.005, 0.005, 0.005, 0.03, 0.05]
    y = [0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.02]
    tcp = np.array([[0.0, yy, zz] for yy, zz in zip(y, z, strict=True)])
    ph = phases(tcp, 30.0, push_z=0.012, low_z=0.025, push_vy=0.01)
    assert ph.tolist()[:3] == ["approach", "approach", "descend"]
    assert "push" in ph.tolist() and ph.tolist()[-1] == "after"


def _rows():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(2):
        step = np.abs(rng.normal(0, 0.01, (100, 6)))
        step[:, 5] = 0
        rows.append({"step": step, "follow": step * 2, "chunk": np.arange(100) % 10 == 0,
                     "phase": np.array(["approach", "descend", "push", "after"] * 25)})  # fmt: skip
    return rows


def test_summarize_and_propose():
    stats = summarize(_rows(), list(ARM6), 30.0)
    assert stats["steps"] == 200
    assert stats["step_chunk_start"]["shoulder_pan"]["n"] == 20
    g = {"max_speed": 0.5, "max_follow_err": 0.05}
    prop = propose(stats, list(ARM6), 0.05, 0.1, 1.2, 1.5, g)
    # the demo limit 0.05 rad/tick dominates the small synthetic steps: 0.05 * 30 * 1.2
    assert prop["max_speed"][:5] == [1.8] * 5 and prop["max_speed"][5] == 0.5
    assert np.isclose(prop["max_follow_err"][0], round(1.8 * (0.1 + 1 / 30) * 2, 3))
    assert "提案" in to_md(stats, list(ARM6), prop)


def test_gate_sim_config_loads_six_joints():
    cfg = load_gate_config(ROOT / "configs" / "gate_sim.yaml")
    assert len(cfg.max_speed) == len(ARM6) == len(cfg.max_follow_err)
    raw = yaml.safe_load((ROOT / "configs" / "gate_sim.yaml").read_text())
    real = yaml.safe_load((ROOT / "configs" / "gate.yaml").read_text())
    for k in ("stale_s", "min_conf", "slow_alpha", "resume_hold_s", "future_tol_s", "dt_max"):
        assert raw[k] == real[k], k  # only G8 / G9 differ from the real-robot config
