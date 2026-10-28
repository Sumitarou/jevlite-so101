"""How fast the policy's commands move, and how far the arm lags behind them (M4a-1).

    uv run --project train python train/cmd_stats.py \
        outputs/sim/20261011_213843_m3d_eval_ck100k/bags

Reads every episode bag of a closed-loop evaluation: /demo/action (the 6-joint command sent each
tick) and /joint_states. Per joint it reports
  step       |command[k] - command[k-1]| (one 30 Hz tick), in rad; p50 / p95 / p99 / max
  follow     |command[k-1] - q_now(t_k)|, the gate's G9 measure (previous command vs the joint
             state when the next command goes out), in rad
and splits the steps by phase (from the commanded TCP: approach / descend / push / after) and by
whether the tick started a new chunk (ticks.jsonl `inferred`). These numbers set max_speed and
max_follow_err in configs/gate_sim.yaml. Writes outputs/sim/<stamp>_m4a_cmd_stats/ (stats.json,
stats.md, steps.png).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from units import ARM6  # noqa: E402

PHASES = ("approach", "descend", "push", "after")


def phases(tcp: np.ndarray, rate_hz: float, push_z: float, low_z: float, push_vy: float):
    """Phase of each tick from the commanded TCP (N, 3) [m, z above the white area].

    push: low (z < push_z) and moving +y faster than push_vy. Before the first push tick: descend
    when z < low_z, else approach. From the last push tick on: after (lift and return)."""
    n = len(tcp)
    vy = np.gradient(tcp[:, 1]) * rate_hz if n > 1 else np.zeros(n)
    push = (tcp[:, 2] < push_z) & (vy > push_vy)
    out = np.where(tcp[:, 2] < low_z, "descend", "approach").astype(object)
    idx = np.flatnonzero(push)
    if len(idx):
        out[idx[0] : idx[-1] + 1] = "push"
        out[idx[-1] + 1 :] = "after"
    return out


def pct(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max())}  # fmt: skip


def per_joint(x: np.ndarray, names: list[str]) -> dict:
    return {n: pct(x[:, j]) for j, n in enumerate(names)}


def episode_rows(t_a, action, t_s, state, inferred):
    """Steps (M-1, J), follow errors (M-1, J) and the chunk-start flag of each step."""
    q_now = np.stack([np.interp(t_a, t_s, state[:, j]) for j in range(state.shape[1])], 1)
    step = np.abs(np.diff(action, axis=0))
    follow = np.abs(action[:-1] - q_now[1:])
    chunk = np.asarray(inferred[1:], bool) if inferred is not None else np.zeros(len(step), bool)
    return step, follow, chunk


def summarize(rows: list[dict], names: list[str], rate_hz: float) -> dict:
    step = np.concatenate([r["step"] for r in rows])
    follow = np.concatenate([r["follow"] for r in rows])
    chunk = np.concatenate([r["chunk"] for r in rows])
    phase = np.concatenate([r["phase"] for r in rows])
    out = {
        "episodes": len(rows),
        "steps": int(len(step)),
        "rate_hz": rate_hz,
        "step": per_joint(step, names),
        "step_chunk_start": per_joint(step[chunk], names),
        "step_within_chunk": per_joint(step[~chunk], names),
        "step_by_phase": {p: per_joint(step[phase == p], names) for p in PHASES},
        "follow": per_joint(follow, names),
        "largest_steps": [],
    }
    norm = step.max(axis=1)
    for k in np.argsort(norm)[::-1][:10]:
        out["largest_steps"].append({
            "rad": float(norm[k]), "joint": names[int(step[k].argmax())],
            "phase": str(phase[k]), "chunk_start": bool(chunk[k]),
        })  # fmt: skip
    return out


def propose(stats: dict, names: list[str], demo_step: float, tau: float, margin: float,
            follow_margin: float, gripper: dict) -> dict:  # fmt: skip
    """max_speed[j] = max(demo limit, policy p99 step) * rate * margin;
    max_follow_err[j] = max(max_speed[j] * (tau + 1/rate) * 2, measured max * follow_margin)."""
    rate = stats["rate_hz"]
    speed, follow = [], []
    for n in names:
        if n == "gripper":
            speed.append(float(gripper["max_speed"]))
            follow.append(float(gripper["max_follow_err"]))
            continue
        s = max(demo_step, stats["step"][n]["p99"]) * rate * margin
        f = max(s * (tau + 1.0 / rate) * 2.0, stats["follow"][n]["max"] * follow_margin)
        speed.append(round(s, 2))
        follow.append(round(f, 3))
    return {"max_speed": speed, "max_follow_err": follow}


def to_md(stats: dict, names: list[str], prop: dict) -> str:
    def table(title, d, scale=1e3, unit="mrad"):
        head = f"| 関節 | p50 | p95 | p99 | 最大 |（{unit}）"
        lines = [f"### {title}", "", head, "|---|---|---|---|---|"]
        for n in names:
            s = d[n]
            if s.get("n", 0) == 0:
                lines.append(f"| {n} | – | – | – | – |")
                continue
            vals = " | ".join(f"{s[k] * scale:.1f}" for k in ("p50", "p95", "p99", "max"))
            lines.append(f"| {n} | {vals} |")
        return "\n".join(lines) + "\n"

    md = [f"# 方策の指令の速さ（{stats['episodes']} 本、{stats['steps']} 刻み、"
          f"{stats['rate_hz']} Hz）\n"]
    md.append(table("1 刻みの変化（全体）", stats["step"]))
    md.append(table("1 刻みの変化（chunk の切り替わりの刻み）", stats["step_chunk_start"]))
    md.append(table("1 刻みの変化（chunk の中）", stats["step_within_chunk"]))
    for p in PHASES:
        md.append(table(f"1 刻みの変化（段階：{p}）", stats["step_by_phase"][p]))
    md.append(table("追従誤差 |前回の指令 − q_now|（G9 の見方）", stats["follow"]))
    md.append("### 大きい 1 刻みの変化（上位 10）\n")
    md.append("| mrad | 関節 | 段階 | chunk の切り替わり |\n|---|---|---|---|")
    for r in stats["largest_steps"]:
        md.append(f"| {r['rad'] * 1e3:.1f} | {r['joint']} | {r['phase']} | {r['chunk_start']} |")
    md.append("\n### 提案（configs/gate_sim.yaml）\n")
    md.append(f"- max_speed [rad/s]: {prop['max_speed']}")
    md.append(f"- max_follow_err [rad]: {prop['max_follow_err']}")
    return "\n".join(md) + "\n"


def plot(rows: list[dict], names: list[str], path: Path, rate_hz: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = np.concatenate([r["step"] for r in rows])
    phase = np.concatenate([r["phase"] for r in rows])
    chunk = np.concatenate([r["chunk"] for r in rows])
    arm = [n for n in names if n != "gripper"]
    fig, axes = plt.subplots(1, len(arm), figsize=(3.2 * len(arm), 3.6), sharey=True,
                             layout="constrained")  # fmt: skip
    for ax, n in zip(axes, arm, strict=True):
        j = names.index(n)
        data = [step[(phase == p) & ~chunk, j] * 1e3 for p in PHASES] + [step[chunk, j] * 1e3]
        ax.boxplot(data, whis=(1, 99), showfliers=True, flierprops={"markersize": 2})
        ax.set_xticks(range(1, len(PHASES) + 2), [*PHASES, "chunk\nstart"], fontsize=7)
        ax.set_title(n, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"|Δ command| per tick [mrad] ({rate_hz:g} Hz)")
    fig.suptitle("policy command step by phase (whiskers p1–p99)", fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("bags", type=Path, help="bags/ folder of an eval run")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "gate_sim.yaml")
    args = ap.parse_args(argv)

    from bagio import episode_dirs, read_episode
    from scripts.sim.outputs import new_run_dir
    from scripts.sim.robustness_core import TcpFK, load_configs

    _, sim_cfg = load_configs()
    tcfg = yaml.safe_load((ROOT / "configs" / "train_act.yaml").read_text())["dataset"]
    scfg = yaml.safe_load(args.config.read_text())["derive"]
    rate = float(sim_cfg["demo"]["rate_hz"])
    fk = TcpFK(sim_cfg)
    names = list(ARM6)
    rows = []
    for bag in episode_dirs(args.bags):
        ep = read_episode(bag, tcfg["topics"], names, with_images=False)
        tj = bag / "ticks.jsonl"
        ticks = [json.loads(s) for s in tj.read_text().splitlines()] if tj.exists() else None
        aligned = ticks is not None and len(ticks) == len(ep.action)
        inferred = [t["inferred"] for t in ticks] if aligned else None
        step, follow, chunk = episode_rows(ep.t_action, ep.action, ep.t_state, ep.state, inferred)
        tcp = fk(ep.action[:, :5])
        ph = phases(tcp, rate, scfg["push_z"], scfg["low_z"], scfg["push_vy"])[1:]
        rows.append({"bag": bag.name, "step": step, "follow": follow, "chunk": chunk, "phase": ph,
                     "inferred_aligned": inferred is not None})  # fmt: skip
        print(f"{bag.name}: {len(ep.action)} commands, max step {step.max() * 1e3:.1f} mrad, "
              f"max follow {follow.max() * 1e3:.1f} mrad", flush=True)  # fmt: skip
    stats = summarize(rows, names, rate)
    stats["source"] = str(args.bags)
    stats["inferred_aligned"] = all(r["inferred_aligned"] for r in rows)
    prop = propose(stats, names, float(sim_cfg["demo"]["max_joint_step"]), scfg["tau_s"],
                   scfg["speed_margin"], scfg["follow_margin"], scfg["gripper"])  # fmt: skip
    stats["proposal"] = prop
    out = new_run_dir("m4a_cmd_stats")
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out / "stats.md").write_text(to_md(stats, names, prop))
    plot(rows, names, out / "steps.png", rate)
    print(json.dumps(prop))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
