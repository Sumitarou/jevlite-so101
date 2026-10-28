"""Tables and figures of the M3e robustness evaluation (no ROS, no GPU; can be re-run any time).

    uv run python scripts/sim/robustness_report.py outputs/sim/<stamp>_m3e_start \
        outputs/sim/<stamp>_m3e_perturb [--name m3e_report]

Reads each folder's summary.json and the per-tick records (bags/<episode>/ticks.jsonl), and
recomputes the metrics (scripts/sim/robustness_core.py) with the metrics settings stored in that
run's summary (or --config). Writes a new folder outputs/sim/<stamp>_<name>/ (never
overwrites) with:
  start.csv / start.md      one row per start condition (success, TCP x vs block x, touched)
  perturb.csv / perturb.md  one row per perturbation x n_action_steps (re-aim time, success)
  report.json               the rows, the x slopes (all / inside / each side outside the demo
                            range) and the inputs
  fig1_start_x.png          block start x vs TCP x while pushing, with the fits inside and
                            outside the demo range (slope 1 = follows the block)
  fig2_start_gap.png        |TCP x - block x| while pushing, per start condition
  fig3_perturb.png          re-aim time and success, perturbation x n_action_steps
  fig4_trace.png            |TCP x - block x| over time for one perturbation, per setting
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.sim.outputs import BASE, new_run_dir  # noqa: E402
from scripts.sim.robustness_core import TcpFK, episode_metrics, x_slopes  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
GOOD, BAD = "#0ca30c", "#d03b3b"  # status: success / failure (always with a marker or label)


def load_run(folder: Path, sim_cfg: dict, mcfg_override: dict | None, fk: TcpFK) -> dict:
    summary = json.loads((folder / "summary.json").read_text())
    mcfg = mcfg_override or summary["config"]["metrics"]
    eps = []
    for line in (folder / "results.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("status") != "done":
            continue
        tf = folder / "bags" / e["bag"] / "ticks.jsonl"
        ticks = [json.loads(t) for t in tf.read_text().splitlines()]
        e["metrics"] = episode_metrics(ticks, fk, sim_cfg, mcfg)
        e["_ticks"] = ticks
        eps.append(e)
    return {"folder": folder, "summary": summary, "episodes": eps, "metrics_cfg": mcfg}


def cm(v, nd=2):
    return None if v is None else round(v * 100, nd)


def start_rows(eps: list[dict]) -> list[dict]:
    rows = []
    for e in eps:
        m, r, (x, y, yaw) = e["metrics"], e["result"], e["block_start"]
        out = [k for k, v in e["outside_demo_range"].items() if v]
        rows.append({
            "condition": e["condition"], "x_cm": cm(x), "y_cm": cm(y),
            "yaw_deg": round(math.degrees(yaw), 1), "outside": "+".join(out) or "-",
            "success": r["success"], "error_cm": cm(r["error_m"]),
            "push_ticks": m["push_ticks"], "tcp_x_push_cm": cm(m["tcp_x_low"]),
            "gap_abs_mean_cm": cm(m["gap_abs_mean"]), "gap_abs_max_cm": cm(m["gap_abs_max"]),
            "touched": m["touched"], "block_moved_cm": cm(m["block_moved_m"]),
            "block_yaw_change_deg": m["block_yaw_change_deg"],
        })  # fmt: skip
    return rows


def perturb_rows(eps: list[dict]) -> list[dict]:
    rows = []
    for e in eps:
        m, r = e["metrics"], e["result"]
        p = m.get("perturb") or {}
        pert = e.get("perturbation") or {}
        rows.append({
            "condition": e["condition"], "perturbation": e["name"],
            "n_action_steps": e["n_action_steps"], "phase": pert.get("phase") or "-",
            "t_event_s": p.get("t_event"),
            "first_inference_after_s": p.get("first_inference_after_s"),
            "perturb_ok": p.get("perturb_ok"), "gap_after_cm": cm(p.get("gap_after")),
            "reaim_s": p.get("reaim_s"), "gap_end_cm": cm(p.get("gap_end")),
            "gap_abs_mean_cm": cm(m["gap_abs_mean"]), "success": r["success"],
            "error_cm": cm(r["error_m"]), "final_yaw_deg": r["yaw_deg"],
        })  # fmt: skip
    return rows


def write_table(rows: list[dict], stem: Path) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return "-" if v is None else str(v)

    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    stem.with_suffix(".md").write_text("\n".join(md) + "\n")


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def _ok_marks(ax, xs, ys, ok):
    xs, ys, ok = np.asarray(xs), np.asarray(ys), np.asarray(ok, bool)
    ax.scatter(xs[ok], ys[ok], s=64, c=GOOD, marker="o", edgecolors=SURF, linewidths=2,
               zorder=3, label="success")  # fmt: skip
    ax.scatter(xs[~ok], ys[~ok], s=64, c=BAD, marker="X", edgecolors=SURF, linewidths=1.5,
               zorder=3, label="failure")  # fmt: skip


def fig_start_x(eps, slopes, demo, frame_half, path):
    import matplotlib.pyplot as plt

    sel = [e for e in eps if e["name"] == "center" or e["name"].startswith("x")]
    sel = [e for e in sel if e["metrics"]["tcp_x_low"] is not None]
    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor=SURF)
    _style(ax, "Block start x vs TCP x while pushing", "block start x [cm]",
           "TCP x while pushing (mean) [cm]")  # fmt: skip
    lim = 6.0
    ax.axvspan(demo["start_x"][0] * 100, demo["start_x"][1] * 100, color=GRID, alpha=0.6, lw=0)
    ax.text(demo["start_x"][0] * 100 + 0.1, lim - 0.4, "demo range", color=INK2, fontsize=8)
    for s in (-1, 1):
        ax.axvline(s * frame_half * 100, color=INK2, lw=1, ls=":")
    ax.text(frame_half * 100 + 0.1, -lim + 0.3, "frame edge", color=INK2, fontsize=8)
    g = np.linspace(-lim, lim, 2)
    ax.plot(g, g, color=INK2, lw=1, ls="--", label="follows the block (slope 1)")
    # each fit is drawn only over the x range of its own points (no line across the whole plot)
    outside = [slopes.get(k) for k in ("outside_minus", "outside_plus")]
    fits = [(slopes.get("inside"), SERIES[0], "fit inside the demo range")]
    fits += [(f, SERIES[1], "fit outside") for f in outside]
    labelled = set()
    for f, color, name in fits:
        if not f:
            continue
        xs = np.array(f["x_range_m"]) * 100
        if name == "fit outside":
            label = "fit outside: slope " + " / ".join(f"{o['slope']:.2f}" for o in outside if o)
        else:
            label = f"{name}: slope {f['slope']:.2f}"
        ax.plot(xs, f["slope"] * xs + f["intercept_m"] * 100, color=color, lw=2,
                label=None if name in labelled else label)  # fmt: skip
        labelled.add(name)
    if sel:
        _ok_marks(ax, [e["block_start"][0] * 100 for e in sel],
                  [e["metrics"]["tcp_x_low"] * 100 for e in sel],
                  [e["result"]["success"] for e in sel])  # fmt: skip
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def fig_start_gap(rows, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.2), facecolor=SURF)
    _style(ax, "|TCP x - block x| while pushing, per start condition", "mean |gap| [cm]", "")
    ys = np.arange(len(rows))[::-1]
    vals = [r["gap_abs_mean_cm"] if r["gap_abs_mean_cm"] is not None else 0.0 for r in rows]
    ax.barh(ys, vals, height=0.6, color=SERIES[0])
    vmax = max(vals + [1.0])
    for y, v, r in zip(ys, vals, rows, strict=True):
        res = "OK" if r["success"] else "FAIL"
        note = "no push" if r["gap_abs_mean_cm"] is None else f"{v:.1f}"
        touched = "touched" if r["touched"] else "not touched"
        ax.text(v + vmax * 0.02, y, f"{note}  {res}  {touched}  moved {r['block_moved_cm']} cm",
                va="center", fontsize=7.5, color=GOOD if r["success"] else BAD)  # fmt: skip
    ax.set_yticks(ys, [r["condition"] for r in rows])
    ax.set_xlim(0, vmax * 2.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def fig_perturb(rows, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = list(dict.fromkeys(r["perturbation"] for r in rows))
    ns = sorted({r["n_action_steps"] for r in rows}, reverse=True)
    by = {(r["perturbation"], r["n_action_steps"]): r for r in rows}
    reaim = np.full((len(names), len(ns)), np.nan)
    for i, a in enumerate(names):
        for j, n in enumerate(ns):
            r = by.get((a, n))
            if r and r["reaim_s"] is not None and a != "baseline":
                reaim[i, j] = r["reaim_s"]
    ramp = ["#cde2fb", "#2a78d6", "#0d366b"]  # sequential blue, light -> dark
    cmap = LinearSegmentedColormap.from_list("blue", ramp).with_extremes(bad=GRID)
    fig, ax = plt.subplots(figsize=(1.9 * len(ns) + 2.6, 0.62 * len(names) + 1.4),
                           facecolor=SURF)  # fmt: skip
    _style(ax, "Re-aim time after the block is moved [s] (grey: none) and success",
           "n_action_steps", "")  # fmt: skip
    vmax = np.nanmax(reaim) if np.isfinite(reaim).any() else 1.0
    im = ax.imshow(np.ma.masked_invalid(reaim), cmap=cmap, vmin=0, vmax=max(vmax, 0.1),
                   aspect="auto")  # fmt: skip
    for i, a in enumerate(names):
        for j, n in enumerate(ns):
            r = by.get((a, n))
            if r is None:
                ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color=INK2)
                continue
            top = "-" if a == "baseline" else ("no re-aim" if r["reaim_s"] is None
                                                else f"{r['reaim_s']:.2f} s")  # fmt: skip
            res = "OK" if r["success"] else "FAIL"
            if a != "baseline" and r["perturb_ok"] is False:
                res += " (move?)"
            dark = np.isfinite(reaim[i, j]) and reaim[i, j] > 0.55 * vmax
            ax.text(j, i, f"{top}\n{res}", ha="center", va="center", fontsize=8,
                    color=SURF if dark else INK)  # fmt: skip
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=8, colors=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def fig_trace(eps, name, fk, tol, path):
    import matplotlib.pyplot as plt

    sel = sorted([e for e in eps if e["name"] == name], key=lambda e: -e["n_action_steps"])
    if not sel:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURF)
    _style(ax, f"|TCP x - block x| over time: {name} (ticks below: inferences)",
           "sim time from the start of the closed loop [s]", "|gap| [cm]")  # fmt: skip
    ymax = 0.0
    for k, e in enumerate(sel):
        tk = e["_ticks"]
        t = np.array([x["t"] for x in tk])
        tcp = fk(np.array([x["state"] for x in tk]))
        gap = np.abs(tcp[:, 0] - np.array([x["block"][0] for x in tk])) * 100
        ymax = max(ymax, float(gap.max()))
        c = SERIES[k % len(SERIES)]
        ax.plot(t, gap, color=c, lw=2, label=f"n_action_steps {e['n_action_steps']}")
        inf = t[[x["inferred"] for x in tk]]
        ax.plot(inf, np.full(len(inf), -0.35 - 0.3 * k), "|", color=c, ms=8, mew=1.5)
        ev = e["metrics"].get("perturb")
        if ev and k == 0:
            ax.axvline(ev["t_event"], color=INK2, lw=1, ls="--")
            ax.text(ev["t_event"] + 0.05, 0.1, "block moved", color=INK2, fontsize=8)
    ax.axhline(tol * 100, color=INK2, lw=1, ls=":")
    ax.text(t[-1], tol * 100 + 0.05, "re-aim tolerance", color=INK2, fontsize=8, ha="right")
    ax.set_ylim(-0.35 - 0.3 * len(sel), ymax * 1.1 + 0.2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3e robustness report")
    ap.add_argument("folders", type=Path, nargs="+", help="outputs/sim/<stamp>_m3e_* folders")
    ap.add_argument("--config", type=Path, help="metrics from this robustness.yaml instead")
    ap.add_argument("--name", default="m3e_report")
    ap.add_argument("--out-base", type=Path, default=BASE, help="parent of the output folder")
    args = ap.parse_args(argv)
    sim_cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    mover = yaml.safe_load(args.config.read_text())["metrics"] if args.config else None
    fk = TcpFK(sim_cfg)
    runs = {}
    for f in args.folders:
        run = load_run(f, sim_cfg, mover, fk)
        runs[run["summary"]["suite"]] = run
    out = new_run_dir(args.name, base=args.out_base)
    report = {"inputs": {s: str(r["folder"]) for s, r in runs.items()},
              "metrics_cfg": {s: r["metrics_cfg"] for s, r in runs.items()}}  # fmt: skip
    start_eps = runs["start"]["episodes"] if "start" in runs else []
    if start_eps:
        rows = start_rows(start_eps)
        write_table(rows, out / "start")
        slopes = x_slopes(start_eps, sim_cfg["demo"])
        report.update(start=rows, x_slope=slopes["all"], x_slopes=slopes)
        fig_start_x(start_eps, slopes, sim_cfg["demo"], sim_cfg["target_frame"]["inner"] / 2,
                    out / "fig1_start_x.png")  # fmt: skip
        fig_start_gap(rows, out / "fig2_start_gap.png")
    if "perturb" in runs:
        eps = list(runs["perturb"]["episodes"])
        # the n_action_steps = 100 baseline is the start suite's centre (same start, no move)
        base = [e for e in start_eps if e["name"] == "center"]
        have = {e["n_action_steps"] for e in eps if e["name"] == "baseline"}
        for e in base:
            if e["n_action_steps"] not in have:
                cid = f"baseline_n{e['n_action_steps']}"
                eps.append({**e, "name": "baseline", "condition": cid})
        rows = perturb_rows(eps)
        rows.sort(key=lambda r: (r["perturbation"] != "baseline", r["perturbation"],
                                 -r["n_action_steps"]))  # fmt: skip
        write_table(rows, out / "perturb")
        report["perturb"] = rows
        fig_perturb(rows, out / "fig3_perturb.png")
        cfg = runs["perturb"]["summary"]["config"]
        tol = runs["perturb"]["metrics_cfg"]["reaim_tol"]
        fig_trace(eps, cfg.get("report", {}).get("trace", "push_x+2"), fk, tol,
                  out / "fig4_trace.png")  # fmt: skip
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"x_slopes": report.get("x_slopes"), "out": str(out)}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
