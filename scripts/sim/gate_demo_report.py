"""M4a report: one figure per scenario and a summary table, from a gate_demo_sim.py run folder.

    uv run python scripts/sim/gate_demo_report.py outputs/sim/<stamp>_m4a_gate [--lang ja|en]
        [--compare outputs/sim/<earlier>_m4a_gate]   # M4a-S: same scenarios, before / after

Writes a new folder outputs/sim/<stamp>_m4a_report/ (never into the run folder): <scenario>.png,
summary.md, report.json. The figure has the layout of scripts/sim_gate_demo.py (same colours,
state band, on_resume lines), on the real closed loop:
  1  judge results (when they came out; size = confidence), the truth, requested, gate state
  2  the rod: distance of its footprint from the white area (< 0: over it; the green band shaded)
  3  joint angles (q_target / q_cmd / q_now) of two joints, state colour in the background
  4  TCP speed (forward kinematics of q_now), with the base_1 run as a reference
  5  reason per tick, plus a row for rate_limited (G8)
With --compare, the earlier run's TCP speed is drawn too, and compare.md / the report's
`compare` give per scenario (both runs): success, SLOW time, TCP speed during SLOW against base_1
in the same time window, the max in 0.5 s after leaving SLOW against base_1 in the same window,
and the arm's lag behind the target at the end of SLOW, with the M4a-S pass marks. base_1 comes
from this run, else from the compared run (the policy is deterministic and CONTINUE unchanged).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jevlite.api import Choice
from scripts.sim.gate_loop import slow_stats, tcp_speed
from scripts.sim.intruder import edge_distance
from scripts.sim.outputs import new_run_dir
from scripts.sim.robustness_core import TcpFK, load_configs
from scripts.sim_gate_demo import (
    BG_ALPHA,
    LINE_STYLE,
    STATE_COLORS,
    _pick_lang,
    place_without_overlap,
)

PLOT_JOINTS = ("shoulder_lift", "elbow_flex")  # the joints that move most (M4a-1)
NAMES6 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REASONS = ["ok", "no_decision", "stale", "future", "non_finite_time", "unknown_choice",
           "non_finite_probs", "low_conf", "follow_error", "non_finite_target"]  # fmt: skip
TEXT = {
    "ja": {"judge": "判定（出た時刻、大きさ = 確信度）", "truth": "正解",
           "requested": "要求（G1/G2 適用後）", "state": "ゲート状態", "level": "段階",
           "rod": "棒の足元と白い面の距離 [cm]", "white": "白い面", "band": "緑の帯",
           "angle": "関節角 [rad]", "target": "q_target（方策）",
           "cmd": "q_cmd（ゲート後）", "now": "q_now", "speed": "手先の速さ [cm/s]",
           "base": "基準（base_1）", "prev": "前回（比較の実行）", "reason": "理由",
           "rate_limited": "rate_limited (G8)",
           "time": "時刻 [s]（閉ループの開始から、sim 時間）", "resume": "on_resume",
           "bg": "背景"},
    "en": {"judge": "judge (release time; size = confidence)", "truth": "truth",
           "requested": "requested (after G1/G2)", "state": "gate state", "level": "level",
           "rod": "rod footprint to white area [cm]", "white": "white area", "band": "green band",
           "angle": "joint angle [rad]", "target": "q_target (policy)", "cmd": "q_cmd (gated)",
           "now": "q_now", "speed": "TCP speed [cm/s]", "base": "reference (base_1)",
           "prev": "earlier run",
           "reason": "reason", "rate_limited": "rate_limited (G8)",
           "time": "time [s] (from loop start, sim)", "resume": "on_resume", "bg": "background"},
}  # fmt: skip
ZONE_LEVEL = {"none": Choice.CONTINUE.value, "edge": Choice.SLOW.value, "center": Choice.STOP.value}


def load_run(run: Path) -> list[dict]:
    eps = []
    for line in (run / "results.jsonl").read_text().splitlines():
        e = json.loads(line)
        bag = run / "bags" / e["bag"]
        e["_ticks"] = [json.loads(x) for x in (bag / "ticks.jsonl").read_text().splitlines()]
        jl = bag / "judge.jsonl"
        e["_judge"] = [json.loads(x) for x in jl.read_text().splitlines()] if jl.exists() else []
        eps.append(e)
    return eps


def series(e: dict, fk, sim_cfg: dict) -> dict:
    ticks = e["_ticks"]
    t = np.array([k["t"] for k in ticks], float)
    g = [k["gate"] for k in ticks]
    q_now = np.array([x["q_now"] for x in g], float)
    rod = np.array([k["intruder"] for k in ticks], float)
    has_rod = bool(e.get("keys"))
    q_target = np.array([x["q_target"] for x in g], float)
    speed = tcp_speed(t, fk(q_now[:, :5])) * 100
    state = np.array([x["state"] for x in g])
    rl = [x["rate_limited"] for x in g]
    return {
        "t": t, "g": g, "q_now": q_now, "state": state,
        "slow": slow_stats(t, state, speed, q_target, q_now, rl),
        "q_target": np.array([x["q_target"] for x in g], float),
        "q_cmd": np.array([x["q_cmd"] for x in g], float),
        "speed": tcp_speed(t, fk(q_now[:, :5])) * 100,
        "rod_d": np.array([edge_distance(x, y, sim_cfg) for x, y, _ in rod]) * 100
        if has_rod else None,
    }  # fmt: skip


def figure(e: dict, s: dict, base: dict | None, sim_cfg: dict, lang: str, prev: dict | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tx = TEXT[lang]
    t, g = s["t"], s["g"]
    state = np.array([Choice[x["state"]].value for x in g])
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [2.2, 1.4, 2.4, 1.6, 1.4]})  # fmt: skip
    ax_s, ax_p, ax_q, ax_v, ax_r = axes
    for ax in (ax_q, ax_v):
        for c in Choice:
            ax.fill_between(t, 0, 1, where=state == c.value, transform=ax.get_xaxis_transform(),
                            color=STATE_COLORS[c], alpha=BG_ALPHA, step="post", lw=0)  # fmt: skip

    # 1: judge / truth / requested / state
    if e["_judge"]:
        jt = np.array([j["t_release"] for j in e["_judge"]])
        jc = [Choice[j["choice"]] for j in e["_judge"]]
        conf = np.array([j["conf"] for j in e["_judge"]])
        ax_s.scatter(jt, [c.value for c in jc], s=8 + 50 * (conf - 0.5).clip(0),
                     c=[STATE_COLORS[c] for c in jc], alpha=0.55, label=tx["judge"],
                     zorder=3)  # fmt: skip
    truth = [ZONE_LEVEL[x["truth"] or "none"] for x in g]
    ax_s.step(t, truth, where="post", color="#3a6ea5", lw=1, alpha=0.6, label=tx["truth"])
    req = [Choice[x["requested"]].value for x in g]
    ax_s.step(t, req, where="post", color="0.45", lw=1, ls="--", label=tx["requested"])
    ax_s.step(t, state, where="post", color="black", lw=2.2, label=tx["state"])
    ax_s.set_yticks([c.value for c in Choice], [c.name for c in Choice])
    ax_s.set_ylim(-0.4, 2.9)
    ax_s.set_ylabel(tx["level"])
    ax_s.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    r = e["result"]
    ax_s.set_title(f"{e['scenario']}: {e.get('desc', '')} — "
                   f"{'成功' if r['success'] else '失敗'} ({r['error_m'] * 100:.1f} cm)"
                   if lang == "ja" else
                   f"{e['scenario']} — {'success' if r['success'] else 'failure'} "
                   f"({r['error_m'] * 100:.1f} cm)")  # fmt: skip

    # 2: the rod
    band = sim_cfg["table"]["band_width"] * 100
    ax_p.axhspan(-5, 0, color="0.85", lw=0, label=tx["white"])
    ax_p.axhspan(0, band, color="#52b96b", alpha=0.35, lw=0, label=tx["band"])
    if s["rod_d"] is not None:
        ax_p.plot(t, s["rod_d"], color="#c0392b", lw=1.6)
    ax_p.set_ylim(-3, 15)
    ax_p.set_ylabel(tx["rod"], fontsize=8)
    ax_p.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)

    # 3: joints
    colors = ["#1f77b4", "#9467bd"]
    for idx, name in enumerate(PLOT_JOINTS):
        j, col = NAMES6.index(name), colors[idx]
        ax_q.plot(t, s["q_now"][:, j], color=col, **LINE_STYLE["now"],
                  label=f"{name} {tx['now']}", solid_capstyle="butt")  # fmt: skip
        ax_q.plot(t, s["q_target"][:, j], color=col, **LINE_STYLE["target"],
                  label=f"{name} {tx['target']}")  # fmt: skip
        ax_q.step(t, s["q_cmd"][:, j], where="post", color=col, **LINE_STYLE["cmd"],
                  label=f"{name} {tx['cmd']}")  # fmt: skip
    handles, labels = ax_q.get_legend_handles_labels()
    for c in Choice:
        handles.append(Patch(facecolor=STATE_COLORS[c], alpha=BG_ALPHA * 2, edgecolor="none"))
        labels.append(f"{tx['bg']}: {c.name}")
    ax_q.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                frameon=False)  # fmt: skip
    ax_q.set_ylabel(tx["angle"])

    # 4: TCP speed
    if base is not None and base is not s:
        ax_v.plot(base["t"], base["speed"], color="0.55", lw=1, ls="--", label=tx["base"])
    if prev is not None:
        ax_v.plot(prev["t"], prev["speed"], color="#c0392b", lw=1, ls=":", alpha=0.8,
                  label=tx["prev"])  # fmt: skip
    ax_v.plot(t, s["speed"], color="black", lw=1.2, label=tx["speed"])
    ax_v.set_ylabel(tx["speed"])
    ax_v.set_ylim(0, max(15.0, float(np.percentile(s["speed"], 99.5)) * 1.1))
    ax_v.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)

    # 5: reason + rate_limited
    used = [r_ for r_ in REASONS if any(x["reason"] == r_ for x in g)]
    pos = {r_: k for k, r_ in enumerate(used)}
    ax_r.scatter(t, [pos[x["reason"]] for x in g], s=6, c="0.25", marker="|")
    lim = np.array([x["rate_limited"] for x in g], bool)
    ax_r.scatter(t[lim], np.full(lim.sum(), len(used)), s=6, c="#3a6ea5", marker="|")
    ax_r.set_yticks(range(len(used) + 1), used + [tx["rate_limited"]], fontsize=8)
    ax_r.set_ylim(-0.6, len(used) + 0.6)
    ax_r.set_ylabel(tx["reason"])
    ax_r.set_xlabel(tx["time"])
    ax_r.set_xlim(0, t[-1])

    texts = []
    for n, k in enumerate(k for k, x in enumerate(g) if x["resumed"]):
        for ax in axes:
            ax.axvline(t[k], color="#3a6ea5", lw=1.4, ls="-.", zorder=4)
        texts.append(ax_s.text(t[k] + 0.08, 2.55, f"{tx['resume']} #{n + 1}", fontsize=8,
                               color="#3a6ea5", va="center",
                               bbox={"fc": "white", "ec": "none", "pad": 1}))  # fmt: skip
    fig.canvas.draw()
    place_without_overlap(texts, fig, base_y=2.55, step_y=-0.3, levels=3)
    return fig


def row(e: dict, s: dict, base: dict | None) -> dict:
    m = e["metrics"]
    intr = m.get("intrusion") or {}
    out = {
        "scenario": e["scenario"], "desc": e.get("desc", ""),
        "success": e["result"]["success"], "error_cm": round(e["result"]["error_m"] * 100, 2),
        "STOP_s": m["state_s"]["STOP"], "SLOW_s": m["state_s"]["SLOW"],
        "resumes": m["resumes"], "reasons": m["reasons"],
        "rate_limited_frac_active": m["rate_limited_frac_active"],
        "follow_error_ticks": m["follow_error_ticks"],
        "judge_age_max_ms": None if m["decision_age_s_max"] is None
        else round(m["decision_age_s_max"] * 1e3, 1),
        "tcp_max_cm_s": round(m["tcp_speed_max"] * 100, 2),
        "tcp_max_intrusion_cm_s": None, "base_same_window_cm_s": None,
        "tcp_max_center_settled_cm_s": None,
        "stop_after_center_s": intr.get("stop_after_center_s"),
        "resume_after_center_s": intr.get("resume_after_center_s"),
        "intruder_meas_err_max_mm": round(e.get("intruder_meas_err_max_m", 0) * 1e3, 2),
    }  # fmt: skip
    if intr:
        out["tcp_max_intrusion_cm_s"] = round(intr["tcp_speed_max"] * 100, 2)
        if intr.get("tcp_speed_max_center_settled") is not None:
            settled = intr["tcp_speed_max_center_settled"]
            out["tcp_max_center_settled_cm_s"] = round(settled * 100, 2)
        if base is not None:
            win = np.zeros(len(base["t"]), bool)
            for a, b in intr["spans"]:
                win |= (base["t"] >= a) & (base["t"] < b)
            if win.any():
                out["base_same_window_cm_s"] = round(float(base["speed"][win].max()), 2)
    return out


def _cell(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "○" if v else "×"
    return str(v)


def to_md(rows: list[dict], run: Path) -> str:
    cols = [("scenario", "シナリオ"), ("success", "成否"), ("error_cm", "誤差 cm"),
            ("STOP_s", "STOP s"), ("SLOW_s", "SLOW s"), ("resumes", "再開"),
            ("tcp_max_intrusion_cm_s", "侵入中の手先 最大 cm/s"),
            ("base_same_window_cm_s", "基準の同じ時間帯 cm/s"),
            ("tcp_max_center_settled_cm_s", "白い面で静定後 cm/s"),
            ("stop_after_center_s", "center→STOP s"), ("resume_after_center_s", "center 後→再開 s"),
            ("rate_limited_frac_active", "rate_limited の割合"),
            ("judge_age_max_ms", "判定の古さ 最大 ms")]  # fmt: skip
    lines = [f"# M4a ゲート付きの閉ループ（{run}）", "",
             "| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]  # fmt: skip
    for r in rows:
        vals = [_cell(r[k]) for k, _ in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## reason の内訳（刻みの数）", ""]
    lines += [f"- {r['scenario']}: {r['reasons']}" for r in rows]
    return "\n".join(lines) + "\n"


def base_window(base: dict, a: float, b: float, how: str) -> float | None:
    """base_1's TCP speed (mean or max) over a <= t <= b, clipped to its own length."""
    win = (base["t"] >= a) & (base["t"] <= b)
    if not win.any():
        return None
    v = base["speed"][win]
    return float(v.mean() if how == "mean" else v.max())


def slow_compare(s: dict, base: dict | None) -> dict:
    """SLOW figures of one run against base_1 in the same time windows."""
    sl = s["slow"]
    out = {"slow_s": sl["slow_s"]}
    if not sl["slow_s"]:
        return out
    mask = s["state"] == "SLOW"
    t = s["t"]
    out.update(speed_mean=round(sl["speed_mean"], 3),
               lag_end_mrad=round(sl["lag_end_max"] * 1e3, 1),
               exit_max=None if sl["exit_speed_max"] is None else round(sl["exit_speed_max"], 3),
               rate_limited_frac=sl["rate_limited_frac"])  # fmt: skip
    if base is not None:
        tb = t[mask & (t <= base["t"][-1])]
        bm = [base["speed"][np.argmin(np.abs(base["t"] - x))] for x in tb]
        if bm:
            sm = float(s["speed"][mask & (t <= base["t"][-1])].mean())
            out["base_mean"] = round(float(np.mean(bm)), 3)
            out["mean_ratio"] = round(sm / max(float(np.mean(bm)), 1e-9), 3)
        ratios = []
        for e_t in sl["exits"]:
            bmax = base_window(base, e_t + 1e-9, e_t + 0.5, "max")
            vmax = float(s["speed"][(t > e_t) & (t <= e_t + 0.5)].max())
            if bmax:
                ratios.append(vmax / bmax)
        out["exit_ratio"] = round(max(ratios), 3) if ratios else None
    return out


def compare_rows(eps, ser, prev_eps, prev_ser, base) -> list[dict]:
    prev = {e["scenario"]: e for e in prev_eps}
    rows = []
    for e in eps:
        k = e["scenario"]
        if k not in prev:
            continue
        new, old = slow_compare(ser[k], base), slow_compare(prev_ser[k], base)
        r = {"scenario": k, "success_old": prev[k]["result"]["success"],
             "success_new": e["result"]["success"],
             "error_cm_old": round(prev[k]["result"]["error_m"] * 100, 2),
             "error_cm_new": round(e["result"]["error_m"] * 100, 2),
             "old": old, "new": new}  # fmt: skip
        checks = {"success": e["result"]["success"] or not prev[k]["result"]["success"]}
        if new.get("mean_ratio") is not None:
            checks["mean_ratio<=0.45"] = new["mean_ratio"] <= 0.45
        if new.get("exit_ratio") is not None:
            checks["exit_ratio<=1.2"] = new["exit_ratio"] <= 1.2
        if new.get("lag_end_mrad") is not None and old.get("lag_end_mrad") is not None:
            checks["lag_end_smaller"] = new["lag_end_mrad"] < old["lag_end_mrad"]
        r["checks"] = checks
        rows.append(r)
    return rows


def compare_md(rows: list[dict], run: Path, prev: Path) -> str:
    def g(d, k, unit=""):
        v = d.get(k)
        return "–" if v is None else f"{v}{unit}"

    lines = [f"# M4a-S 比較：{prev}（前）→ {run}（後）", "",
             "| シナリオ | 成否 前→後 | 誤差 cm 前→後 | SLOW s 前→後 | SLOW 中の平均 / 基準 前→後 "
             "| 抜けた後 0.5 s の最大 / 基準 前→後 | SLOW の終わりの遅れ mrad 前→後 | 目安 |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        o, n = r["old"], r["new"]
        ok = ", ".join(f"{k} {'○' if v else '×'}" for k, v in r["checks"].items())
        lines.append(
            f"| {r['scenario']} | {_cell(r['success_old'])}→{_cell(r['success_new'])} "
            f"| {r['error_cm_old']}→{r['error_cm_new']} | {g(o, 'slow_s')}→{g(n, 'slow_s')} "
            f"| {g(o, 'mean_ratio')}→{g(n, 'mean_ratio')} "
            f"| {g(o, 'exit_ratio')}→{g(n, 'exit_ratio')} "
            f"| {g(o, 'lag_end_mrad')}→{g(n, 'lag_end_mrad')} | {ok} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--lang", default="ja", choices=["ja", "en"])
    ap.add_argument("--out", type=Path, help="default: a new outputs/sim/<stamp>_m4a_report/")
    ap.add_argument("--compare", type=Path, help="an earlier run with the same scenarios")
    args = ap.parse_args(argv)
    _, sim_cfg = load_configs()
    fk = TcpFK(sim_cfg)
    eps = load_run(args.run)
    if not eps:
        print("no episodes", file=sys.stderr)
        return 1
    lang = _pick_lang(args.lang)
    out = args.out or new_run_dir("m4a_report")
    out.mkdir(parents=True, exist_ok=True)
    ser = {e["scenario"]: series(e, fk, sim_cfg) for e in eps}
    prev_eps = load_run(args.compare) if args.compare else []
    prev_ser = {e["scenario"]: series(e, fk, sim_cfg) for e in prev_eps}
    base = ser.get("base_1") or prev_ser.get("base_1")
    rows = []
    import matplotlib.pyplot as plt

    for e in eps:
        s = ser[e["scenario"]]
        fig = figure(e, s, base, sim_cfg, lang, prev=prev_ser.get(e["scenario"]))
        fig.savefig(out / f"{e['scenario']}.png", dpi=120)
        plt.close(fig)
        rows.append(row(e, s, base))
    (out / "summary.md").write_text(to_md(rows, args.run))
    report = {"run": str(args.run), "rows": rows}
    if args.compare:
        comp = compare_rows(eps, ser, prev_eps, prev_ser, base)
        report["compare"] = {"run": str(args.compare), "rows": comp}
        (out / "compare.md").write_text(compare_md(comp, args.run, args.compare))
        print(compare_md(comp, args.run, args.compare))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(to_md(rows, args.run))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
