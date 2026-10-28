"""M4b: compare the image judges (v0 / v1 / v2) on the held-out rod images (uv env, GPU).

    uv run python scripts/eval_sim_judges.py --data data/sim_intruder/<run> \
        --v1 models/judge/<stamp>_v1 --v2 models/judge/<stamp>_v2 [--split test] [--no-v0]

Judges (each through judge(image) -> Decision, one image at a time, as in the loop):
  v0-choices  zero-shot SigLIP with the hand prompts of configs/choices.yaml (v0 as it is)
  v0-zones    zero-shot SigLIP with the rod prompts of configs/judge_sim.yaml `v0.prompts`
  v1, v2      the trained students (jevlite.student.load_student)
A choice is mapped back to a zone with configs/m4a.yaml `zones` (CONTINUE -> none, ...).
Reported per judge: accuracy overall and on the subsets (near a boundary: within data.near_eval_m;
arm near the rod; on the table), the confusion, center -> none (center -> CONTINUE, must be 0),
the confidence of right and wrong answers (and how many wrong ones fall under the gate's min_conf,
where G2 makes them one step safer), and the judge's own latency (Decision.latency_ms) p50 / p95
/ max over at least eval.bench_n calls after eval.warmup.
Output: outputs/sim/<stamp>_m4b_eval/ with summary.md, results.json, predictions.csv,
confidence.png and errors_<judge>.png (the unsafe-side errors first).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from jevlite.api import Choice  # noqa: E402
from jevlite.student import ZONES, load_student, zero_shot_zone_judge  # noqa: E402
from scripts.sim.judge_data import (  # noqa: E402
    labels,
    load_image,
    load_rows,
    rows_digest,
    split_rows,
    subset_masks,
    zone_metrics,
)
from scripts.sim.outputs import git_state, new_run_dir  # noqa: E402

CONFIG = ROOT / "configs" / "judge_sim.yaml"
CONF_BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]


def zone_of_choice(zones: dict[str, str]) -> dict[str, str]:
    inv = {}
    for z, c in zones.items():
        if Choice.parse(c).name in inv:
            raise ValueError("zones must map to distinct choices to read a choice as a zone")
        inv[Choice.parse(c).name] = z
    return inv


def pct(v, q) -> float | None:
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def run_judge(judge, rows, images, warmup: int, bench_n: int) -> dict:
    """Judge every image once (predictions, latency), then repeat over the images until
    bench_n latencies are collected. Warm-up calls are not counted."""
    for i in range(warmup):
        judge.judge(images[i % len(images)])
    preds, lat = [], []
    for im in images:
        d = judge.judge(im)
        preds.append(d)
        lat.append(d.latency_ms)
    i = 0
    while len(lat) < bench_n:
        lat.append(judge.judge(images[i % len(images)]).latency_ms)
        i += 1
    return {"decisions": preds, "latency_ms": lat}


def judge_report(decisions, rows, inv: dict[str, str], masks: dict, min_conf: float,
                 lat: list[float]) -> dict:  # fmt: skip
    y = labels(rows)
    p = np.array([ZONES.index(inv[d.choice.name]) for d in decisions])
    conf = np.array([d.confidence for d in decisions])
    ok = p == y
    under = p < y
    rep = {name: zone_metrics(y[m], p[m]) for name, m in masks.items()}
    hist = {k: np.histogram(conf[sel], bins=CONF_BINS)[0].tolist()
            for k, sel in (("right", ok), ("wrong", ~ok))}  # fmt: skip
    rep["confidence"] = {
        "bins": CONF_BINS[:-1] + [1.0],
        "hist": hist,
        "right_median": pct(conf[ok], 50),
        "wrong_median": pct(conf[~ok], 50),
        "wrong_below_min_conf": int(np.sum(~ok & (conf < min_conf))),
        "unsafe_below_min_conf": int(np.sum(under & (conf < min_conf))),
        "unsafe": int(np.sum(under)),
        "min_conf": min_conf,
    }
    rep["latency_ms"] = {"n": len(lat), "p50": pct(lat, 50), "p95": pct(lat, 95),
                         "p99": pct(lat, 99), "max": round(float(max(lat)), 3)}  # fmt: skip
    rep["_pred"], rep["_conf"] = p, conf
    return rep


def build_judges(args, cfg, zones) -> dict:
    judges = {}
    if not args.no_v0:
        from jevlite.config import load_config
        from jevlite.zeroshot import SiglipBackend, ZeroShotJudge

        s = cfg["siglip"]
        backend = SiglipBackend(s["model_id"], s["revision"], s["device"], s["dtype"])
        judges["v0-choices"] = ZeroShotJudge(backend, load_config("choices.yaml"),
                                             source="v0-choices")  # fmt: skip
        judges["v0-zones"] = zero_shot_zone_judge(backend, cfg["v0"]["prompts"], zones)
    for name in ("v1", "v2"):
        d = getattr(args, name)
        if d is not None:
            judges[name] = load_student(d, cfg, zones)
    return judges


def write_md(path: Path, res: dict, near_m: float) -> None:
    head = ("| 判定器 | 全体 | 境目付近 | アームが近い | 台の上 | none / edge / center | "
            "center→CONTINUE | center→SLOW | edge→CONTINUE | "
            "危険側の誤り（うち確信度 < min_conf）| "
            "確信度の中央値 正解 / 誤り | 遅れ p50 / p95 / 最大 ms |")  # fmt: skip
    lines = [f"# M4b 判定器の比較（{res['split']}、{res['n']} 枚）", "",
             f"境目付近：境目から {near_m * 100:.0f} cm 以内（{res['subset_n']['near']} 枚）。"
             f"アームが近い：{res['subset_n']['arm_near']} 枚。", "", head,
             "|" + "---|" * 12]  # fmt: skip
    for name, r in res["judges"].items():
        a, c = r["all"], r["confidence"]
        rec = " / ".join(str(a["recall"][z]) for z in ZONES)
        lat = r["latency_ms"]
        lines.append(
            f"| {name} | {a['accuracy']} | {r['near']['accuracy']} | {r['arm_near']['accuracy']} "
            f"| {r['on_table']['accuracy']} | {rec} | **{a['center_to_none']}** "
            f"| {a['center_to_edge']} | {a['edge_to_none']} "
            f"| {c['unsafe']}（{c['unsafe_below_min_conf']}）"
            f"| {c['right_median']} / {c['wrong_median']} "
            f"| {lat['p50']} / {lat['p95']} / {lat['max']} |")  # fmt: skip
    lines += ["", "混同行列（行＝正解、列＝判定、none / edge / center の順）", ""]
    for name, r in res["judges"].items():
        lines.append(f"- {name}: {r['all']['confusion']}")
    path.write_text("\n".join(lines) + "\n")


def plot_confidence(path: Path, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res["judges"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.8), squeeze=False)
    labels_x = [f"{a:g}" for a in CONF_BINS[:-1]]
    for ax, name in zip(axes[0], names, strict=True):
        h = res["judges"][name]["confidence"]["hist"]
        x = np.arange(len(labels_x))
        ax.bar(x - 0.2, h["right"], 0.4, label="right", color="#3b7dd8")
        ax.bar(x + 0.2, h["wrong"], 0.4, label="wrong", color="#d8533b")
        ax.set_xticks(x, labels_x, rotation=90, fontsize=7)
        ax.set_yscale("symlog")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("confidence (bin start)", fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def error_sheet(path: Path, rows, images, pred, conf, max_n: int = 24) -> int:
    from PIL import Image, ImageDraw

    y = labels(rows)
    wrong = np.flatnonzero(pred != y)
    order = sorted(wrong, key=lambda i: (pred[i] >= y[i], -conf[i]))[:max_n]  # unsafe first
    if not order:
        return 0
    w, h = 320, 240
    sheet = Image.new("RGB", (w * 6, h * ((len(order) + 5) // 6)))
    for k, i in enumerate(order):
        im = Image.fromarray(images[i]).resize((w, h))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w, 14], fill=(0, 0, 0))
        d.text((3, 2), f"{ZONES[y[i]]}->{ZONES[pred[i]]} {conf[i]:.2f} "
                       f"d={float(rows[i]['d_boundary']) * 100:.1f}cm {rows[i]['file']}",
               fill=(255, 255, 255))  # fmt: skip
        sheet.paste(im, ((k % 6) * w, (k // 6) * h))
    sheet.save(path)
    return len(order)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4b: compare the image judges")
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--v1", type=Path)
    ap.add_argument("--v2", type=Path)
    ap.add_argument("--no-v0", action="store_true")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--name", default="eval")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    zones = yaml.safe_load((ROOT / cfg["zones_config"]).read_text())["zones"]
    inv = zone_of_choice(zones)
    min_conf = float(yaml.safe_load((ROOT / cfg["eval"]["gate_config"]).read_text())["min_conf"])
    near_m = float(cfg["data"]["near_eval_m"])
    rows = split_rows(load_rows(args.data), args.split)
    if not rows:
        ap.error(f"no {args.split} images")
    t0 = time.monotonic()
    images = [load_image(r["path"]) for r in rows]
    print(f"{len(rows)} {args.split} images loaded in {time.monotonic() - t0:.0f} s", flush=True)
    masks = subset_masks(rows, near_m)
    out = new_run_dir(f"m4b_{args.name}")
    res = {"split": args.split, "n": len(rows), "data": [str(d) for d in args.data],
           "data_digest": rows_digest(rows), "git": git_state(),
           "subset_n": {k: int(m.sum()) for k, m in masks.items()},
           "models": {k: str(getattr(args, k)) for k in ("v1", "v2") if getattr(args, k)},
           "judges": {}}  # fmt: skip
    pred_cols = {}
    for name, judge in build_judges(args, cfg, zones).items():
        t1 = time.monotonic()
        r = run_judge(judge, rows, images, int(cfg["eval"]["warmup"]), int(cfg["eval"]["bench_n"]))
        rep = judge_report(r["decisions"], rows, inv, masks, min_conf, r["latency_ms"])
        p, conf = rep.pop("_pred"), rep.pop("_conf")
        pred_cols[name] = (p, conf)
        rep["errors_shown"] = error_sheet(out / f"errors_{name}.png", rows, images, p, conf)
        res["judges"][name] = rep
        a = rep["all"]
        print(f"{name}: acc {a['accuracy']} (near {rep['near']['accuracy']}, arm "
              f"{rep['arm_near']['accuracy']}), center->none {a['center_to_none']}, latency "
              f"p95 {rep['latency_ms']['p95']} ms ({time.monotonic() - t1:.0f} s)", flush=True)
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data", "file", "zone", "d_boundary", "arm_near",
                    *[f"{n}_{k}" for n in pred_cols for k in ("zone", "conf")]])  # fmt: skip
        for i, r in enumerate(rows):
            w.writerow([r["data"], r["file"], r["zone"], r["d_boundary"], r["arm_near"],
                        *[v for p, c in pred_cols.values()
                          for v in (ZONES[p[i]], round(float(c[i]), 4))]])  # fmt: skip
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    write_md(out / "summary.md", res, near_m)
    plot_confidence(out / "confidence.png", res)
    print((out / "summary.md").read_text())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
