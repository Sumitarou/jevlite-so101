"""Hardware-free demo of the safety gate: synthetic decisions + synthetic joint targets.

Feeds a scripted judge timeline (clear -> hand enters -> flicker -> stall -> recovery) and
sinusoidal policy targets through SafetyGate, then plots q_cmd, gate state and on_resume
calls over time. Gate thresholds come from configs/gate.yaml, the scenario from
configs/sim_demo.yaml.

    uv run python scripts/sim_gate_demo.py [--out outputs/sim_gate_demo.png] [--lang ja|en]
        [--no-rate-limit] [--no-follow-check] [--no-csv]

Writes the PNG and a CSV of the timeline next to it. No GUI is opened.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from gate.safety_gate import GateConfig, Reason, SafetyGate, load_gate_config
from jevlite.api import Choice, Decision
from jevlite.config import load_config

JA_FONTS = ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic"]
STATE_COLORS = {Choice.CONTINUE: "#2a9d5c", Choice.SLOW: "#e9a23b", Choice.STOP: "#d1495b"}

TEXT = {
    "en": {
        "title": "Safety gate on a synthetic scenario",
        "judge": "judge output",
        "requested": "requested (after G1/G2)",
        "state": "gate state",
        "level": "level",
        "angle": "joint angle [deg]",
        "target": "q_target",
        "cmd": "q_cmd",
        "now": "q_now",
        "reason": "reason",
        "g8_on": "G8 on",
        "g8_off": "G8 off",
        "g9_on": "G9 on",
        "g9_off": "G9 off",
        "blocked": "blocked",
        "bg": "background",
        "rate_limited": "rate_limited (G8)",
        "time": "time [s]",
        "resume": "on_resume",
        "no_decision": "no decision",
    },
    "ja": {
        "title": "合成シナリオでのセーフティ・ゲート動作",
        "judge": "判定器の出力",
        "requested": "要求（G1/G2 適用後）",
        "state": "ゲート状態",
        "level": "段階",
        "angle": "関節角 [deg]",
        "target": "q_target（方策）",
        "cmd": "q_cmd（ゲート後）",
        "now": "q_now（実機の模擬）",
        "reason": "理由",
        "g8_on": "G8 あり",
        "g8_off": "G8 なし",
        "g9_on": "G9 あり",
        "g9_off": "G9 なし",
        "blocked": "障害物で停止",
        "bg": "背景",
        "rate_limited": "rate_limited (G8)",
        "time": "時刻 [s]",
        "resume": "on_resume",
        "no_decision": "判定なし",
    },
}


@dataclass
class SimResult:
    t: np.ndarray
    q_target: np.ndarray  # (T, J)
    q_now: np.ndarray  # (T, J) position before the step
    q_cmd: np.ndarray  # (T, J)
    state: list[Choice]
    requested: list[Choice]
    reason: list[Reason]
    rate_limited: list[bool] = field(default_factory=list)
    blocked: list[bool] = field(default_factory=list)  # arm held by an obstacle (q_now frozen)
    judge_t: list[float] = field(default_factory=list)  # when the judge emitted
    judge_choice: list[Choice] = field(default_factory=list)
    judge_conf: list[float] = field(default_factory=list)
    resume_times: list[float] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)


def joint_targets(joints: list[dict[str, Any]], t: float) -> np.ndarray:
    return np.array(
        [j["offset"] + j["amp"] * np.sin(2 * np.pi * j["freq_hz"] * t + j["phase"]) for j in joints]
    )


def segment_at(segments: list[dict[str, Any]], t: float) -> dict[str, Any]:
    for seg in segments:
        if t < seg["until"]:
            return seg
    return segments[-1]


def segment_spans(segments: list[dict[str, Any]]) -> list[tuple[float, float, dict]]:
    spans, start = [], 0.0
    for seg in segments:
        spans.append((start, float(seg["until"]), seg))
        start = float(seg["until"])
    return spans


def simulate(gate_cfg: GateConfig, scenario: dict[str, Any]) -> SimResult:
    control_hz, judge_hz = int(scenario["control_hz"]), int(scenario["judge_hz"])
    if control_hz % judge_hz:
        raise ValueError("control_hz must be an integer multiple of judge_hz")
    every = control_hz // judge_hz
    n = int(round(scenario["duration_s"] * control_hz))
    gain = float(scenario["servo_gain"])
    joints = scenario["joints"]

    resume_times: list[float] = []
    now_holder = {"t": 0.0}
    gate = SafetyGate(gate_cfg, on_resume=lambda: resume_times.append(now_holder["t"]))

    res = SimResult(
        t=np.arange(n) / control_hz,
        q_target=np.zeros((n, len(joints))),
        q_now=np.zeros((n, len(joints))),
        q_cmd=np.zeros((n, len(joints))),
        state=[],
        requested=[],
        reason=[],
        resume_times=resume_times,
        joint_names=[j["name"] for j in joints],
    )
    q_now = joint_targets(joints, 0.0)
    latest: Decision | None = None
    tick_in_segment: dict[int, int] = {}

    for k in range(n):
        t = k / control_hz
        now_holder["t"] = t
        if k % every == 0:
            seg = segment_at(scenario["segments"], t)
            if not seg.get("dropout", False):
                i = tick_in_segment.get(id(seg), 0)
                tick_in_segment[id(seg)] = i + 1
                name, conf = seg["pattern"][i % len(seg["pattern"])]
                choice = Choice.parse(name)
                rest = (1.0 - conf) / 2
                probs = {c.name: (conf if c is choice else rest) for c in Choice}
                latest = Decision(
                    choice=choice,
                    probs=probs,
                    t_mono=t + float(seg.get("t_offset_s", 0.0)),
                    latency_ms=0.0,
                    source="sim",
                )
                res.judge_t.append(t)
                res.judge_choice.append(choice)
                res.judge_conf.append(conf)

        q_target = joint_targets(joints, t)
        out = gate.step(latest, t, q_now, q_target)
        res.q_target[k], res.q_now[k], res.q_cmd[k] = q_target, q_now, out.q_cmd
        res.state.append(out.state)
        res.requested.append(out.requested)
        res.reason.append(out.reason)
        res.rate_limited.append(out.rate_limited)
        is_blocked = blocked_at(scenario, t)
        res.blocked.append(is_blocked)
        if not is_blocked:
            q_now = q_now + gain * (out.q_cmd - q_now)  # simple first-order servo
    return res


def blocked_at(scenario: dict[str, Any], t: float) -> bool:
    return any(ob["start"] <= t < ob["end"] for ob in scenario.get("obstacles", []))


def max_jump(res: SimResult) -> float:
    """Largest change of the command between consecutive steps while moving (STOP excluded:
    stopping snaps the command back to q_now on purpose)."""
    jumps = [
        np.abs(res.q_cmd[k] - res.q_cmd[k - 1]).max()
        for k in range(1, len(res.t))
        if res.state[k] is not Choice.STOP
    ]
    return float(max(jumps, default=0.0))


def compute_metrics(res: SimResult, scenario: dict[str, Any]) -> dict[str, Any]:
    """Comparison metrics for the report table (definitions in configs/sim_demo.yaml)."""
    m_cfg = scenario["metrics"]
    t = res.t
    w0, w1 = m_cfg["steady_window"]
    steady = (t >= w0) & (t < w1)
    detect_t = None
    for ob in scenario.get("obstacles", []):
        hits = [
            float(t[k])
            for k in range(len(t))
            if ob["start"] <= t[k] < ob["end"] and res.reason[k] is Reason.FOLLOW_ERROR
        ]
        if hits:
            detect_t = hits[0] - ob["start"]
    unnecessary = restops = 0
    for r in res.resume_times:
        after = [k for k in range(len(t)) if t[k] > r and res.state[k] is Choice.STOP]
        if after and t[after[0]] - r <= m_cfg["restop_window_s"]:
            restops += 1
            # a STOP the judge asked for with a valid decision (reason ok) is not unnecessary
            if res.reason[after[0]] is not Reason.OK:
                unnecessary += 1
    return {
        "max_cmd_step": max_jump(res),
        "steady_track_err": float(np.abs(res.q_target - res.q_now)[steady].mean()),
        "obstacle_detected": detect_t is not None,
        "detect_time_s": detect_t,
        "unnecessary_stops": unnecessary,
        "restops_within_window": restops,
        "continue_ratio": float(np.mean([s is Choice.CONTINUE for s in res.state])),
    }


def write_csv(res: SimResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["t", "state", "requested", "reason", "rate_limited", "blocked", "on_resume"]
            + [f"{n}_{kind}" for n in res.joint_names for kind in ("target", "now", "cmd")]
        )
        resumes = set(res.resume_times)
        for k, t in enumerate(res.t):
            row = [f"{t:.3f}", res.state[k].name, res.requested[k].name, str(res.reason[k])]
            row += [int(res.rate_limited[k]), int(res.blocked[k]), int(float(t) in resumes)]
            for j in range(len(res.joint_names)):
                row += [f"{res.q_target[k, j]:.3f}", f"{res.q_now[k, j]:.3f}"]
                row.append(f"{res.q_cmd[k, j]:.3f}")
            w.writerow(row)


def _pick_lang(lang: str) -> str:
    if lang != "ja":
        return lang
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in JA_FONTS:
        if name in available:
            import matplotlib

            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            return "ja"
    print("warning: no Japanese font found; falling back to English labels", file=sys.stderr)
    return "en"


LABEL_BASE_Y, LABEL_STEP_Y, LABEL_LEVELS = 2.3, 0.28, 3  # segment names above STOP


def place_without_overlap(
    texts,
    fig,
    pad_px: float = 6.0,
    base_y: float = LABEL_BASE_Y,
    step_y: float = LABEL_STEP_Y,
    levels: int = LABEL_LEVELS,
) -> list[int]:
    """Move each text one row (step_y) at a time until it no longer overlaps an earlier one.

    Uses the rendered extents, so it works for any label length or language.
    Returns the row chosen for each text (0 = base_y).
    """
    renderer = fig.canvas.get_renderer()
    placed: list[tuple[int, Any]] = []
    rows = []
    for text in texts:
        for row in range(levels):
            text.set_y(base_y + step_y * row)
            bb = text.get_window_extent(renderer).expanded(1.0, 1.0)
            bb = bb.from_extents(bb.x0 - pad_px, bb.y0, bb.x1 + pad_px, bb.y1)
            if not any(r == row and bb.overlaps(other) for r, other in placed):
                break
        placed.append((row, bb))
        rows.append(row)
    return rows


BG_ALPHA = 0.14  # state colour band behind the joint-angle panel
LINE_STYLE = {  # q_target thin/dashed/light, q_now a wide translucent band, q_cmd crisp on top
    "target": {"ls": "--", "lw": 1.0, "alpha": 0.55},
    "now": {"ls": "-", "lw": 4.5, "alpha": 0.28},
    "cmd": {"ls": "-", "lw": 1.5, "alpha": 1.0},
}


def plot(
    res: SimResult,
    scenario: dict[str, Any],
    out_png: Path,
    lang: str = "ja",
    rate_limit: bool = True,
    follow_check: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    fig = make_figure(res, scenario, lang, rate_limit, follow_check)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def make_figure(
    res: SimResult,
    scenario: dict[str, Any],
    lang: str = "ja",
    rate_limit: bool = True,
    follow_check: bool = True,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lang = _pick_lang(lang)
    tx = TEXT[lang]
    levels = [c.value for c in Choice]
    level_names = [c.name for c in Choice]

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 8.5), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [2.2, 3, 1.6]},
    )  # fmt: skip
    ax_s, ax_q, ax_r = axes

    # background: scenario segments + state colour band
    seg_texts = []
    for start, end, seg in segment_spans(scenario["segments"]):
        for ax in axes:
            ax.axvline(start, color="0.85", lw=0.8, zorder=0)
        label = seg.get("label", {})
        seg_texts.append(
            ax_s.text(
                (start + end) / 2, LABEL_BASE_Y, label.get(lang, label.get("en", "")),
                ha="center", va="bottom", fontsize=9, color="0.3",
            )
        )  # fmt: skip
    state_vals = np.array([s.value for s in res.state])
    for c in Choice:
        ax_q.fill_between(
            res.t, 0, 1, where=state_vals == c.value, transform=ax_q.get_xaxis_transform(),
            color=STATE_COLORS[c], alpha=BG_ALPHA, step="post", lw=0,
        )  # fmt: skip

    # panel 1: judge output, requested, gate state
    jc = np.array([c.value for c in res.judge_choice])
    ax_s.scatter(
        res.judge_t, jc, s=12 + 60 * (np.array(res.judge_conf) - 0.5).clip(0),
        c=[STATE_COLORS[c] for c in res.judge_choice], alpha=0.6, label=tx["judge"], zorder=3,
    )  # fmt: skip
    req = np.array([c.value for c in res.requested])
    ax_s.step(res.t, req, where="post", color="0.45", lw=1, ls="--", label=tx["requested"])
    ax_s.step(res.t, state_vals, where="post", color="black", lw=2.2, label=tx["state"])
    ax_s.set_yticks(levels, level_names)
    ax_s.set_ylim(-0.4, LABEL_BASE_Y + LABEL_STEP_Y * LABEL_LEVELS + 0.05)
    ax_s.set_ylabel(tx["level"])
    ax_s.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)

    # panel 2: joint angles
    colors = ["#1f77b4", "#9467bd", "#8c564b", "#17becf"]
    for idx, name in enumerate(scenario.get("plot_joints", res.joint_names[:2])):
        j = res.joint_names.index(name)
        col = colors[idx % len(colors)]
        ax_q.plot(res.t, res.q_now[:, j], color=col, **LINE_STYLE["now"],
                  label=f"{name} {tx['now']}", solid_capstyle="butt")  # fmt: skip
        ax_q.plot(res.t, res.q_target[:, j], color=col, **LINE_STYLE["target"],
                  label=f"{name} {tx['target']}")  # fmt: skip
        ax_q.step(res.t, res.q_cmd[:, j], where="post", color=col, **LINE_STYLE["cmd"],
                  label=f"{name} {tx['cmd']}")  # fmt: skip
    for ob in scenario.get("obstacles", []):
        ax_q.axvspan(ob["start"], ob["end"], facecolor="none", edgecolor="0.35", hatch="//",
                     lw=0, zorder=1)  # fmt: skip
        ax_q.text(
            (ob["start"] + ob["end"]) / 2, 0.02, tx["blocked"],
            transform=ax_q.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="0.25",
            bbox={"fc": "white", "ec": "none", "pad": 1, "alpha": 0.8},
        )  # fmt: skip
    ax_q.set_ylabel(tx["angle"])
    handles, labels = ax_q.get_legend_handles_labels()
    for c in Choice:
        handles.append(Patch(facecolor=STATE_COLORS[c], alpha=BG_ALPHA * 2, edgecolor="none"))
        labels.append(f"{tx['bg']}: {c.name}")
    if scenario.get("obstacles"):
        handles.append(Patch(facecolor="none", edgecolor="0.35", hatch="//", lw=0))
        labels.append(tx["blocked"])
    ax_q.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                frameon=False)  # fmt: skip

    # panel 3: reason (+ a row for G8 activity)
    reasons = list(Reason)
    ridx = np.array([reasons.index(r) for r in res.reason])
    used = sorted(set(ridx.tolist()))
    pos = {r: i for i, r in enumerate(used)}
    ax_r.scatter(res.t, [pos[r] for r in ridx], s=6, c="0.25", marker="|")
    row_g8 = len(used)
    limited_t = res.t[np.array(res.rate_limited, dtype=bool)]
    ax_r.scatter(limited_t, np.full(len(limited_t), row_g8), s=6, c="#3a6ea5", marker="|")
    ax_r.set_yticks(
        range(len(used) + 1), [str(reasons[r]) for r in used] + [tx["rate_limited"]], fontsize=8
    )
    ax_r.set_ylim(-0.6, len(used) + 0.6)
    ax_r.set_ylabel(tx["reason"])
    ax_r.set_xlabel(tx["time"])

    # on_resume markers across all panels
    resume_texts = []
    for i, tr in enumerate(res.resume_times):
        for ax in axes:
            ax.axvline(tr, color="#3a6ea5", lw=1.4, ls="-.", zorder=4)
        resume_texts.append(
            ax_s.text(
                tr + 0.04, 1.6, f"{tx['resume']} #{i + 1}", ha="left", va="center",
                fontsize=8, color="#3a6ea5", bbox={"fc": "white", "ec": "none", "pad": 1},
                zorder=5,
            )
        )  # fmt: skip

    g8 = tx["g8_on" if rate_limit else "g8_off"]
    g9 = tx["g9_on" if follow_check else "g9_off"]
    ax_s.set_title(f"{tx['title']} — {g8} / {g9}")
    ax_r.set_xlim(0, res.t[-1])
    fig.canvas.draw()  # run the constrained layout so text extents are final
    place_without_overlap(seg_texts, fig)
    place_without_overlap(resume_texts, fig, base_y=1.6, step_y=-0.25, levels=4)
    return fig


def summarize(res: SimResult) -> str:
    jump = max_jump(res)
    lines = [f"on_resume called {len(res.resume_times)}x at " +
             ", ".join(f"{t:.2f}s" for t in res.resume_times),
             f"max command step while moving = {jump:.2f} (rate limited on "
             f"{sum(res.rate_limited)}/{len(res.rate_limited)} steps)"]  # fmt: skip
    prev = None
    for k, s in enumerate(res.state):
        if s is not prev:
            lines.append(f"  {res.t[k]:5.2f}s  -> {s.name:<8} (requested {res.requested[k].name}, "
                         f"reason {res.reason[k]})")  # fmt: skip
            prev = s
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/sim_gate_demo.png"))
    ap.add_argument("--lang", choices=["ja", "en"], default="ja")
    ap.add_argument("--scenario", default="sim_demo.yaml", help="file name under configs/")
    ap.add_argument(
        "--no-rate-limit", action="store_true", help="disable G8 (for comparison figures)"
    )
    ap.add_argument(
        "--no-follow-check", action="store_true", help="disable G9 (for comparison figures)"
    )
    ap.add_argument("--no-csv", action="store_true", help="do not write the CSV next to the PNG")
    args = ap.parse_args(argv)

    scenario = load_config(args.scenario)
    gate_cfg = load_gate_config()
    if args.no_rate_limit:
        gate_cfg = without_rate_limit(gate_cfg)
    if args.no_follow_check:
        gate_cfg = without_follow_check(gate_cfg)
    res = simulate(gate_cfg, scenario)
    png = plot(
        res,
        scenario,
        args.out,
        args.lang,
        rate_limit=not args.no_rate_limit,
        follow_check=not args.no_follow_check,
    )
    print(summarize(res))
    print(f"-> {png}")
    if not args.no_csv:
        csv_path = args.out.with_suffix(".csv")
        write_csv(res, csv_path)
        print(f"-> {csv_path}")
    return 0


def without_rate_limit(cfg: GateConfig) -> GateConfig:
    """Same gate with G8 effectively disabled (huge max_speed), for comparison figures."""
    return replace(cfg, max_speed=(1e12,) * len(cfg.max_speed))


def without_follow_check(cfg: GateConfig) -> GateConfig:
    """Same gate with G9 effectively disabled (huge max_follow_err), for comparison figures."""
    return replace(cfg, max_follow_err=(1e12,) * len(cfg.max_follow_err))


if __name__ == "__main__":
    sys.exit(main())
