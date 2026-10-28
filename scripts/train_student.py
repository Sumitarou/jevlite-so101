"""M4b: train the image judges v1 / v2 on the labelled rod images (uv env, GPU).

    uv run python scripts/train_student.py v1 --data data/sim_intruder/<run> [<run> ...]
    uv run python scripts/train_student.py v1 --data ... --features pooled grid2 --name grid
    uv run python scripts/train_student.py v2 --data data/sim_intruder/<run> [<run> ...]

Settings: configs/judge_sim.yaml (crop, siglip, v1, v2). Training uses the train split, model
selection the val split; the test split is not touched (scripts/eval_sim_judges.py).
  v1  SigLIP features (frozen; cached as features_<digest>.npz in the first data folder) ->
      StandardScaler + LogisticRegression for each C; the best on val is kept.
  v2  MobileNetV3-Small from ImageNet weights; inputs are cropped and resized once on the GPU with
      the judge's own preprocessing (jevlite.student.GpuPre), then augmented per batch
      (brightness / contrast jitter, shift; no flips). The best epoch on val is kept.
Output: models/judge/<stamp>_<v1|v2>[_name]/ (git-ignored) with meta.json (settings, data digest,
model revision, val metrics) and head.npz (v1) or model.pt (v2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from jevlite.student import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    ZONES,
    GpuPre,
    LinearHead,
    SiglipFeatures,
    build_mobilenet,
    resolve_device,
)
from scripts.sim.judge_data import (  # noqa: E402
    better,
    labels,
    load_image,
    load_rows,
    rows_digest,
    split_rows,
    zone_metrics,
)
from scripts.sim.outputs import git_state, new_run_dir  # noqa: E402

CONFIG = ROOT / "configs" / "judge_sim.yaml"


def batches(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def siglip_features(rows, cfg, cache_dir: Path) -> np.ndarray:
    v1 = cfg["v1"]
    key = rows_digest(rows) + "|" + json.dumps([cfg["siglip"], cfg["crop"], v1["features"]])
    import hashlib

    path = cache_dir / f"features_{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    if path.exists():
        print(f"features from {path}", flush=True)
        return np.load(path)["feats"]
    fx = SiglipFeatures(cfg["siglip"], cfg["crop"], v1["features"])
    out, t0 = [], time.monotonic()
    for sl in batches(len(rows), int(v1["batch"])):
        out.append(fx([load_image(r["path"]) for r in rows[sl]]))
    feats = np.concatenate(out).astype(np.float32)
    np.savez(path, feats=feats)
    print(f"features {feats.shape} in {time.monotonic() - t0:.0f} s -> {path}", flush=True)
    return feats


def train_v1(rows, cfg) -> tuple[dict, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    v1 = cfg["v1"]
    feats = siglip_features(rows, cfg, Path(rows[0]["path"]).parents[1])
    idx = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s])
           for s in ("train", "val")}  # fmt: skip
    y = labels(rows)
    sc = StandardScaler().fit(feats[idx["train"]])
    xt, xv = sc.transform(feats[idx["train"]]), sc.transform(feats[idx["val"]])
    best, best_m, tried = None, None, []
    for c in v1["C"]:
        lr = LogisticRegression(C=float(c), class_weight=v1["class_weight"],
                                max_iter=int(v1["max_iter"]))  # fmt: skip
        lr.fit(xt, y[idx["train"]])
        m = zone_metrics(y[idx["val"]], lr.predict(xv))
        m_train = zone_metrics(y[idx["train"]], lr.predict(xt))
        tried.append({"C": c, "val": m, "train_accuracy": m_train["accuracy"]})
        print(f"C {c}: train {m_train['accuracy']}, val {m['accuracy']}, "
              f"val center->none {m['center_to_none']}", flush=True)  # fmt: skip
        if better(m, best_m):
            best, best_m = (c, lr), m
    c, lr = best
    if list(lr.classes_) != list(range(len(ZONES))):
        raise RuntimeError(f"classes {lr.classes_} (a zone is missing from train)")
    head = LinearHead(sc.mean_, sc.scale_, lr.coef_, lr.intercept_)
    meta = {"kind": "v1", "features": v1["features"], "siglip": cfg["siglip"], "C": c,
            "class_weight": v1["class_weight"], "tried": tried, "val": best_m}  # fmt: skip
    return meta, {"head": head}


def load_inputs(rows, pre: GpuPre, batch: int):
    """Cropped + resized uint8 inputs (N, 3, h, w) on the CPU, made with the judge's GpuPre."""
    import torch

    out = []
    for sl in batches(len(rows), batch):
        x = pre.resized([load_image(r["path"]) for r in rows[sl]])
        out.append(x.round().clamp(0, 255).to(torch.uint8).cpu())
    return torch.cat(out)


def augment(x01, rng, jitter: float, shift: int):
    """Per-sample brightness / contrast jitter and a shift (edges repeated). x01 in [0, 1]."""
    import torch

    n = x01.shape[0]
    dev = x01.device
    b = torch.from_numpy(rng.uniform(1 - jitter, 1 + jitter, n)).float().to(dev).view(n, 1, 1, 1)
    c = torch.from_numpy(rng.uniform(1 - jitter, 1 + jitter, n)).float().to(dev).view(n, 1, 1, 1)
    m = x01.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x01 - m) * c + m) * b
    if shift > 0:
        pad = torch.nn.functional.pad(x, (shift,) * 4, mode="replicate")
        h, w = x.shape[2:]
        dx, dy = rng.integers(0, 2 * shift + 1, n), rng.integers(0, 2 * shift + 1, n)
        x = torch.stack([pad[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n)])
    return x.clamp(0, 1)


def train_v2(rows, cfg) -> tuple[dict, dict]:
    import torch

    v2 = cfg["v2"]
    torch.manual_seed(int(v2["seed"]))
    rng = np.random.default_rng(int(v2["seed"]))
    dev = resolve_device("auto")
    pre = GpuPre(cfg["crop"], v2["input"], IMAGENET_MEAN, IMAGENET_STD, dev, torch.float32)
    tr, va = split_rows(rows, "train"), split_rows(rows, "val")
    t0 = time.monotonic()
    xt, xv = load_inputs(tr, pre, int(v2["batch"])), load_inputs(va, pre, int(v2["batch"]))
    yt, yv = torch.from_numpy(labels(tr)), labels(va)
    print(f"inputs train {tuple(xt.shape)}, val {tuple(xv.shape)} "
          f"in {time.monotonic() - t0:.0f} s", flush=True)  # fmt: skip
    net = build_mobilenet(v2["arch"], v2["weights"]).to(dev)
    counts = np.bincount(yt.numpy(), minlength=len(ZONES))
    w = torch.tensor(len(yt) / (len(ZONES) * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w.to(dev))
    opt = torch.optim.AdamW(net.parameters(), lr=float(v2["lr"]),
                            weight_decay=float(v2["weight_decay"]))  # fmt: skip
    epochs, bs = int(v2["epochs"]), int(v2["batch"])
    steps = epochs * ((len(yt) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(v2["lr"]), total_steps=steps)

    def predict(x):
        net.eval()
        out = []
        with torch.inference_mode():
            for sl in batches(len(x), 256):
                out.append(net(pre.normalize(x[sl].to(dev).float() / 255.0)).argmax(1).cpu())
        return torch.cat(out).numpy()

    best_m, best_state, hist = None, None, []
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(yt)))
        tl, t0 = 0.0, time.monotonic()
        for sl in batches(len(yt), bs):
            b = perm[sl]
            x = augment(xt[b].to(dev).float() / 255.0, rng, float(v2["jitter"]),
                        int(v2["shift_px"]))  # fmt: skip
            loss = loss_fn(net(pre.normalize(x)), yt[b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss) * len(b)
        m = zone_metrics(yv, predict(xv))
        hist.append({"epoch": ep + 1, "loss": round(tl / len(yt), 5), "val": m,
                     "wall_s": round(time.monotonic() - t0, 1)})  # fmt: skip
        print(f"epoch {ep + 1}: loss {tl / len(yt):.4f}, val {m['accuracy']}, "
              f"center->none {m['center_to_none']} ({time.monotonic() - t0:.0f} s)", flush=True)
        if better(m, best_m):
            best_m = m
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_ep = ep + 1
    meta = {"kind": "v2", "arch": v2["arch"], "weights": v2["weights"], "input": v2["input"],
            "train": {k: v2[k] for k in ("epochs", "batch", "lr", "weight_decay", "jitter",
                                         "shift_px", "seed")},
            "best_epoch": best_ep, "history": hist, "val": best_m}  # fmt: skip
    return meta, {"state": best_state}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4b: train the image judge v1 / v2")
    ap.add_argument("kind", choices=["v1", "v2"])
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="data/sim_intruder/<run>")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--name", default="", help="output folder suffix")
    ap.add_argument("--features", nargs="+", help="v1: override v1.features (pooled, grid2)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    if args.features:
        cfg["v1"]["features"] = args.features
    rows = load_rows(args.data)
    counts = {s: len(split_rows(rows, s)) for s in ("train", "val", "test")}
    if not counts["train"] or not counts["val"]:
        ap.error(f"train and val need images, got {counts}")
    print(f"{len(rows)} images {counts}", flush=True)
    t0 = time.monotonic()
    meta, art = (train_v1 if args.kind == "v1" else train_v2)(rows, cfg)
    out = new_run_dir(args.kind + (f"_{args.name}" if args.name else ""),
                      base=ROOT / "models" / "judge")  # fmt: skip
    if args.kind == "v1":
        art["head"].save(out / "head.npz")
    else:
        import torch

        torch.save(art["state"], out / "model.pt")
    meta.update({"source": f"{args.kind}:{out.name}", "crop": cfg["crop"], "zones_order": ZONES,
                 "data": [str(d) for d in args.data], "data_digest": rows_digest(rows),
                 "counts": counts, "git": git_state(),
                 "train_wall_s": round(time.monotonic() - t0, 1)})  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=list) + "\n")
    print(json.dumps({"val": meta["val"], "out": str(out.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
