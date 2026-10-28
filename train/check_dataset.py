"""Check a converted LeRobotDataset (M3a): counts, alignment, normalization stats, video loss, plot.

    uv run --project train python train/check_dataset.py          # settings: configs/train_act.yaml
    uv run --project train python train/check_dataset.py --root /tmp/x

Writes to outputs/sim/<stamp>_m3a_check/:
  report.json        episode/frame counts, alignment summary, per-dimension mean/std/min/max of
                     observation.state and action (dimensions with std < check.min_std are flagged:
                     lerobot's MEAN_STD normalization divides by std + 1e-8), video vs raw image
                     error
  stats.md           the same tables in Markdown
  episode<N>.png     camera frames at the phase starts above the 6 joints (state and action)
  video_worst.png    the frame with the largest video error: raw | decoded | |diff| x8
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

from bagio import read_episode  # noqa: E402
from scripts.sim.bag_align import align  # noqa: E402
from scripts.sim.outputs import new_run_dir  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_STATE, C_ACTION = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def dim_stats(ds, key: str, names: list[str], min_std: float) -> list[dict]:
    s = ds.meta.stats[key]
    rows = []
    for j, n in enumerate(names):
        std = float(np.asarray(s["std"])[j])
        rows.append(
            {
                "dim": n,
                "mean": float(np.asarray(s["mean"])[j]),
                "std": std,
                "min": float(np.asarray(s["min"])[j]),
                "max": float(np.asarray(s["max"])[j]),
                "tiny_std": std < min_std,
                # what a 0.002 rad offset becomes after (x - mean) / (std + 1e-8)
                "z_of_2mrad": 0.002 / (std + 1e-8),
            }
        )
    return rows


def episode_range(ds, ep: int) -> tuple[int, int]:
    e = ds.meta.episodes[ep]
    return int(e["dataset_from_index"]), int(e["dataset_to_index"])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def video_error(ds, cfg: dict, side: dict, ep: int, out: Path) -> dict:
    """Decoded video frames vs the raw rgb8 images of the bag, for every frame of one episode."""
    row = side[ep]
    e = read_episode(ROOT / cfg["src"] / row["bag"], cfg["topics"], cfg["joints"])
    al = align(e.t_action, e.t_state, e.state, e.t_image, hold_state=cfg["hold_state"])
    keep = np.flatnonzero(al.keep)
    lo, hi = episode_range(ds, ep)
    if hi - lo != len(keep):
        raise ValueError(f"episode {ep}: {hi - lo} frames in the dataset, {len(keep)} in the bag")
    key = cfg["image_key"]
    mean_abs, max_abs, p99, ps, act_err = [], [], [], [], 0.0
    worst = (-1.0, None, None, None)
    for f, k in enumerate(keep):
        item = ds[lo + f]
        dec = item[key].numpy()  # (3, H, W) uint8 with return_uint8
        dec = np.transpose(dec, (1, 2, 0))
        raw = e.images[al.image_index[k]]
        d = np.abs(dec.astype(np.int16) - raw.astype(np.int16))
        mean_abs.append(float(d.mean()))
        max_abs.append(int(d.max()))
        p99.append(float(np.percentile(d, 99)))
        ps.append(psnr(raw, dec))
        act_err = max(act_err, float(np.abs(item["action"].numpy() - e.action[k]).max()))
        if mean_abs[-1] > worst[0]:
            worst = (mean_abs[-1], raw, dec, f)
    import matplotlib.pyplot as plt

    _, raw, dec, f = worst
    d = np.abs(dec.astype(np.int16) - raw.astype(np.int16)).astype(np.float64)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURF)
    for a, img, t in zip(
        ax,
        (raw, dec, np.clip(d * 8, 0, 255).astype(np.uint8)),
        ("raw (bag, rgb8)", "decoded (dataset video)", "|diff| x 8"),
        strict=True,
    ):
        a.imshow(img)
        a.set_title(t, color=INK, fontsize=11)
        a.axis("off")
    fig.suptitle(
        f"episode {ep}, frame {f}: mean |diff| {d.mean():.2f}, PSNR {psnr(raw, dec):.1f} dB",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "video_worst.png", dpi=100, facecolor=SURF)
    plt.close(fig)
    return {
        "episode": ep,
        "frames": len(keep),
        "mean_abs_mean": float(np.mean(mean_abs)),
        "mean_abs_max": float(np.max(mean_abs)),
        "max_abs_max": int(np.max(max_abs)),
        "p99_abs_mean": float(np.mean(p99)),
        "p99_abs_max": float(np.max(p99)),
        "psnr_mean_db": float(np.mean(ps)),
        "psnr_min_db": float(np.min(ps)),
        "worst_frame": int(f),
        "action_readback_max_err": act_err,  # float32 rounding only
    }


def plot_episode(ds, cfg: dict, side: dict, ep: int, out: Path) -> Path:
    import matplotlib.pyplot as plt

    lo, hi = episode_range(ds, ep)
    items = [ds[i] for i in range(lo, hi)]
    t = np.array([float(it["timestamp"]) for it in items])
    st = np.stack([it["observation.state"].numpy() for it in items])
    ac = np.stack([it["action"].numpy() for it in items])
    info = json.loads((ROOT / cfg["src"] / side[ep]["bag"] / "episode.json").read_text())
    first = side[ep]["first_action_tick"]  # dataset frame 0 = this action tick
    phases = [(n, max(a - first, 0)) for n, a, _ in info["phases"] if a - first < len(items)]
    shown = [p for p in phases if p[0] in ("to_start", "before_push", "push", "after_push",
                                           "settle", "to_home")]  # fmt: skip
    names = cfg["joints"]
    fig = plt.figure(figsize=(16, 11), facecolor=SURF)
    gs = fig.add_gridspec(4, len(shown), height_ratios=[0.8, 1, 1, 1], hspace=0.3)
    for c, (n, f) in enumerate(shown):
        a = fig.add_subplot(gs[0, c])
        a.imshow(np.transpose(items[f][cfg["image_key"]].numpy(), (1, 2, 0)))
        a.set_title(f"{n}\nframe {f} ({t[f]:.2f} s)", color=INK, fontsize=9)
        a.axis("off")
    sub = gs[1:, :].subgridspec(3, 2, hspace=0.35, wspace=0.15)
    for j, n in enumerate(names):
        a = fig.add_subplot(sub[j // 2, j % 2])
        a.set_facecolor(SURF)
        for _, f in phases:
            a.axvline(t[f], color=GRID, lw=1, zorder=0)
        a.plot(t, st[:, j], color=C_STATE, lw=2, label="observation.state")
        a.plot(t, ac[:, j], color=C_ACTION, lw=2, ls=(0, (4, 2)), label="action")
        a.set_title(f"{n} [rad]", color=INK, fontsize=10, loc="left")
        a.tick_params(colors=INK2, labelsize=8)
        a.ticklabel_format(axis="y", useOffset=False)
        lo_y, hi_y = a.get_ylim()
        if hi_y - lo_y < 0.02:  # a constant joint must look flat, not zoomed into 1e-7 rad
            mid = (lo_y + hi_y) / 2
            a.set_ylim(mid - 0.01, mid + 0.01)
        a.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        if j == 0:
            a.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="best")
        if j >= len(names) - 2:
            a.set_xlabel("time in episode [s] (lines: phase starts)", color=INK2, fontsize=9)
    fig.suptitle(
        f"episode {ep} ({side[ep]['bag']}): {len(items)} frames at {cfg['fps']} fps; "
        f"result error {info['result']['error_m'] * 100:.1f} cm",
        color=INK,
    )
    path = out / f"episode{ep}.png"
    fig.savefig(path, dpi=90, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return path


def md_table(rows: list[dict]) -> str:
    head = "| dim | mean | std | min | max | z of 0.002 rad | |\n|---|---|---|---|---|---|---|\n"
    return head + "".join(
        f"| {r['dim']} | {r['mean']:+.4f} | {r['std']:.2e} | {r['min']:+.4f} | {r['max']:+.4f} "
        f"| {r['z_of_2mrad']:.3g} | {'**std tiny**' if r['tiny_std'] else ''} |\n"
        for r in rows
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_act.yaml"))
    ap.add_argument("--root", help="override dataset.root")
    args = ap.parse_args()
    full = yaml.safe_load(Path(args.config).read_text())
    cfg, chk = full["dataset"], full["check"]
    root = Path(args.root or ROOT / cfg["root"])

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(cfg["repo_id"], root=root, return_uint8=True)
    side = [json.loads(x) for x in (root / "jevlite_episodes.jsonl").read_text().splitlines()]
    conv = json.loads((root / "jevlite_convert.json").read_text())
    out = new_run_dir("m3a_check")

    item = ds[0]
    readback = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in item.items()
        if hasattr(v, "shape")
    }
    names = list(cfg["joints"])
    stats = {
        "observation.state": dim_stats(ds, "observation.state", names, chk["min_std"]),
        "action": dim_stats(ds, "action", names, chk["min_std"]),
    }
    report = {
        "root": str(root),
        "episodes": ds.meta.total_episodes,
        "frames": ds.meta.total_frames,
        "fps": ds.meta.fps,
        "frames_per_episode": [r["frames"] for r in side],
        "convert": {k: v for k, v in conv.items() if k != "config"},
        "readback_frame0": readback,
        "task": item["task"],
        "stats": stats,
        "normalization_eps": 1e-8,
        "video": video_error(ds, cfg, side, chk["compare_episode"], out),
        "plot": str(plot_episode(ds, cfg, side, chk["plot_episode"], out).name),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "stats.md").write_text(
        "## observation.state\n\n" + md_table(stats["observation.state"])
        + "\n## action\n\n" + md_table(stats["action"])
    )  # fmt: skip
    print(json.dumps({k: v for k, v in report.items() if k != "stats"}, indent=2))
    print((out / "stats.md").read_text())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
