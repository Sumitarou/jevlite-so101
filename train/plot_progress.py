"""Plot a training run's progress.jsonl (loss, step time, VRAM) and print a short summary.

    uv run --project train python train/plot_progress.py models/act/<run> [--name m3c_loss]

Writes outputs/sim/<stamp>_<name>/loss.png and summary.json (never overwrites).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sim.outputs import new_run_dir  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1 = "#2a78d6"


def load(run: Path) -> tuple[list[dict], list[dict]]:
    rows, events = [], []
    for line in (run / "progress.jsonl").read_text().splitlines():
        r = json.loads(line)
        (events if "event" in r else rows).append(r)
    return rows, events


def summarize(rows: list[dict], events: list[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    k = max(1, len(rows) // 10)
    first, last = rows[:k], rows[-k:]

    def mean(rs, key):
        return sum(r[key] for r in rs) / len(rs)

    def median(key):  # robust to the start-up windows (first run and each resume)
        v = sorted(r[key] for r in rows)
        return v[len(v) // 2]

    return {
        "rows": len(rows),
        "steps": [rows[0]["step"], rows[-1]["step"]],
        "loss_first": mean(first, "loss"),
        "loss_last": mean(last, "loss"),
        "l1_loss_first": mean(first, "l1_loss") if "l1_loss" in rows[0] else None,
        "l1_loss_last": mean(last, "l1_loss") if "l1_loss" in rows[0] else None,
        "update_s_median": median("update_s"),
        "dataloading_s_median": median("dataloading_s"),
        "step_s_median": median("update_s") + median("dataloading_s"),
        "vram_max_alloc_mib": max(r.get("vram_max_alloc_mib", 0) for r in rows),
        "vram_reserved_mib": max(r.get("vram_reserved_mib", 0) for r in rows),
        "checkpoints": [e["step"] for e in events if e["event"] == "checkpoint"],
    }


def plot(rows: list[dict], events: list[dict], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURF)
    panels = (
        ("loss", "loss (L1 + KL weight x KL)", True),
        ("step_s", "time per step [s] (data + update)", False),
    )
    for a, (key, label, log) in zip(ax, panels, strict=True):
        ys = [r["update_s"] + r["dataloading_s"] if key == "step_s" else r[key] for r in rows]
        a.set_facecolor(SURF)
        a.plot(steps, ys, color=C1, lw=2)
        for e in events:
            if e["event"] == "checkpoint":
                a.axvline(e["step"], color=GRID, lw=1, zorder=0)
        if log:
            a.set_yscale("log")
        a.set_title(label, color=INK, fontsize=10, loc="left")
        a.set_xlabel("step (vertical lines: checkpoints)", color=INK2, fontsize=9)
        a.tick_params(colors=INK2, labelsize=8)
        a.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor=SURF)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--name", default="m3c_loss")
    args = ap.parse_args()
    rows, events = load(args.run)
    out = new_run_dir(args.name)
    summary = {"run": args.run.name, **summarize(rows, events)}
    plot(rows, events, out / "loss.png", args.run.name)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
