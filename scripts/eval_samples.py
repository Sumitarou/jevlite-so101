"""Stage 4a: evaluate v0 and the Ollama teachers on a labelled capture session.

    uv run python scripts/eval_samples.py --session 20261011-desk \\
        [--models v0,qwen2.5vl:7b,gemma3:4b] [--prompts baseline,def-v1]

Ground truth is the metadata.csv label (none->A->CONTINUE, edge->B->SLOW, center->C->STOP,
configs/eval.yaml). For every model x prompt variant each image is judged once (saved images,
so Decision.t_mono is the judge start time). qwen3.5:27b runs only when named in --models.

Writes to outputs/eval/<session>/ (git-ignored):
    predictions.csv   one row per image x model x prompt (choice, probs, latency_ms, ...)
    summary.json/md   accuracy, confusion matrix (rows = truth, cols = prediction), latency
                      p50/p95, teacher errors, skipped models with the reason
    prompts.md        the full prompt texts / v0 labels that were used
A model that cannot run (Ollama not reachable, model not installed, no torch/GPU for v0) is
skipped and the reason recorded; the rest still run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from jevlite.api import Choice, Judge
from jevlite.config import load_config
from jevlite.samples import Session, load_session, resolve_subsets
from scripts.capture_samples import REPO_ROOT, is_git_ignored

LETTERS = {Choice.CONTINUE: "A", Choice.SLOW: "B", Choice.STOP: "C"}
PRED_COLUMNS = [
    "model", "prompt", "file", "label", "truth", "pred", "truth_letter", "pred_letter",
    "correct", "p_CONTINUE", "p_SLOW", "p_STOP", "latency_ms", "source", "error", "subsets",
    "reason", "object_present", "touches_white", "contradiction",
]  # fmt: skip


# --- prompts ------------------------------------------------------------------------------


def teacher_prompt(name: str, eval_cfg: dict, teacher_cfg: dict) -> str:
    text = eval_cfg["teacher_prompts"][name]
    return teacher_cfg["prompt"] if text is None else text


def v0_choices(name: str, eval_cfg: dict, choices_cfg: dict) -> dict:
    labels = eval_cfg["v0_prompts"][name]
    if labels is None:
        return choices_cfg
    return {"choices": [{"name": c.name, "prompts": list(labels[c.name])} for c in Choice]}


def prompts_markdown(prompt_names, eval_cfg, teacher_cfg, choices_cfg) -> str:
    lines = ["# Prompts used", ""]
    for name in prompt_names:
        lines += [f"## {name}", "", "### Teacher (Ollama)", "", "```text"]
        lines += [teacher_prompt(name, eval_cfg, teacher_cfg).rstrip(), "```", ""]
        mode = (eval_cfg.get("teacher_modes") or {}).get(name)
        if mode:
            lines += [f"Mode: `{json.dumps(mode)}` (attr: the code picks the choice from "
                      "object_present / touches_white)", ""]  # fmt: skip
        lines += ["### v0 text labels (SigLIP)", ""]
        if name not in eval_cfg["v0_prompts"]:
            lines += ["(none for this prompt: v0 is skipped)", ""]
            continue
        for ch in v0_choices(name, eval_cfg, choices_cfg)["choices"]:
            for p in ch["prompts"]:
                lines.append(f"- {ch['name']}: `{p}`")
        lines.append("")
    return "\n".join(lines)


# --- judges ------------------------------------------------------------------------------


class Skip(Exception):
    """A model cannot run here; the message is recorded as the reason."""


def ollama_models(url: str, timeout: float, http_get=httpx.get) -> dict[str, str]:
    """Installed model name -> digest. Raises Skip if Ollama is not reachable."""
    try:
        r = http_get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        r.raise_for_status()
        return {m["name"]: m.get("digest", "") for m in r.json().get("models", [])}
    except Exception as e:
        raise Skip(f"Ollama not reachable at {url}: {type(e).__name__}: {e}") from e


def real_factories(eval_cfg, teacher_cfg, choices_cfg, http_get=httpx.get, references=None):
    """make(model, prompt_name) -> Judge, plus run info (digests / revisions).

    references: prompt name -> reference image (RGB array) for prompts that use one."""
    references = references or {}
    info: dict[str, Any] = {}
    cache: dict[str, Any] = {}

    def make(model: str, prompt_name: str) -> Judge:
        if model == "v0":
            if eval_cfg["v0_prompts"].get(prompt_name, "missing") == "missing":
                raise Skip(f"v0 has no text labels for prompt {prompt_name}")
            if "v0_backend" not in cache:
                try:
                    from jevlite.zeroshot import SiglipBackend

                    z = load_config("zeroshot.yaml")
                    cache["v0_backend"] = SiglipBackend(
                        z["model_id"], z["revision"], z.get("device", "auto"),
                        z.get("dtype", "float16"),
                    )  # fmt: skip
                    info["v0"] = {"model_id": z["model_id"], "revision": z["revision"]}
                except Exception as e:
                    raise Skip(f"v0 cannot load SigLIP: {type(e).__name__}: {e}") from e
            from jevlite.zeroshot import ZeroShotJudge

            z = load_config("zeroshot.yaml")
            return ZeroShotJudge(
                cache["v0_backend"], v0_choices(prompt_name, eval_cfg, choices_cfg),
                softmax_scale=z.get("softmax_scale"),
            )  # fmt: skip
        if "ollama" not in cache:
            try:
                cache["ollama"] = ollama_models(
                    teacher_cfg["ollama_url"], float(eval_cfg["ollama_check_timeout_s"]), http_get
                )
            except Skip as e:
                cache["ollama"] = e
        if isinstance(cache["ollama"], Skip):
            raise cache["ollama"]
        if model not in cache["ollama"]:
            raise Skip(f"Ollama model {model} is not installed")
        info.setdefault("ollama", {})[model] = cache["ollama"][model]
        from jevlite.teacher_ollama import OllamaAttrTeacher, OllamaTeacher

        cfg = {**teacher_cfg, "prompt": teacher_prompt(prompt_name, eval_cfg, teacher_cfg)}
        timeout = (eval_cfg.get("timeout_s_by_model") or {}).get(model)
        if timeout is not None:
            cfg["timeout_s"] = timeout
        mode = (eval_cfg.get("teacher_modes") or {}).get(prompt_name, {})
        if mode.get("mode", "choice") == "attr":
            return OllamaAttrTeacher(cfg, model=model, reference=references.get(prompt_name))
        return OllamaTeacher(cfg, model=model)

    return make, info


# --- evaluation --------------------------------------------------------------------------


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def evaluate(
    session: Session,
    judge: Judge,
    model: str,
    prompt: str,
    label_to_choice: dict,
    warmup: int,
    subsets: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    subsets = subsets or {}
    images = [(s, load_rgb(s.path)) for s in session.samples]
    for _, img in images[: min(warmup, len(images))]:
        judge.judge(img)  # not recorded: first calls include model loading
    rows = []
    for s, img in images:
        d = judge.judge(img)
        truth = Choice[label_to_choice[s.label]]
        rows.append(
            {
                "model": model,
                "prompt": prompt,
                "file": s.file,
                "label": s.label,
                "truth": truth.name,
                "pred": d.choice.name,
                "truth_letter": LETTERS[truth],
                "pred_letter": LETTERS[d.choice],
                "correct": int(d.choice is truth),
                **{f"p_{c.name}": round(float(d.probs.get(c.name, 0.0)), 4) for c in Choice},
                "latency_ms": round(d.latency_ms, 2),
                "source": d.source,
                "error": getattr(judge, "last_error", None) or "",
                "subsets": ";".join(n for n, files in subsets.items() if s.file in files),
                **_extra(getattr(judge, "last_extra", None) or {}),
            }
        )
    return rows


def _extra(extra: dict[str, Any]) -> dict[str, Any]:
    def b(v):
        return "" if v is None else int(bool(v))

    return {
        "reason": extra.get("reason", ""),
        "object_present": b(extra.get("object_present")),
        "touches_white": b(extra.get("touches_white")),
        "contradiction": b(extra.get("contradiction")),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [c.name for c in Choice]
    cm = [[0] * 3 for _ in names]
    for r in rows:
        cm[names.index(r["truth"])][names.index(r["pred"])] += 1
    lat = np.array([r["latency_ms"] for r in rows], dtype=float)
    per_class = {}
    for i, n in enumerate(names):
        total = sum(cm[i])
        per_class[n] = round(cm[i][i] / total, 4) if total else None
    return {
        "n": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None,
        "recall": per_class,
        "confusion": {"rows_truth_cols_pred": names, "matrix": cm},
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2) if len(lat) else None,
            "p95": round(float(np.percentile(lat, 95)), 2) if len(lat) else None,
            "mean": round(float(lat.mean()), 2) if len(lat) else None,
        },
        "teacher_errors": sum(r["source"] == "teacher_error" for r in rows),
        # safety: protected-area images (truth STOP) judged CONTINUE
        "stop_as_continue": cm[names.index("STOP")][names.index("CONTINUE")],
        "contradictions": sum(r.get("contradiction") == 1 for r in rows),
    }


def summarize_groups(rows: list[dict[str, Any]], subset_names: list[str]) -> dict[str, Any]:
    """{"all": ..., <subset>: ..., "other": ...}; only "all" when there are no subsets."""
    out = {"all": summarize(rows)}
    if subset_names:
        for name in subset_names:
            out[name] = summarize([r for r in rows if name in r["subsets"].split(";")])
        out["other"] = summarize([r for r in rows if not r["subsets"]])
    return out


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


GROUP_TITLES = {"all": "全体", "other": "それ以外"}


def summary_markdown(
    session: Session, results: dict, skipped: dict, info: dict, subsets: dict[str, set[str]]
) -> str:
    lines = [
        f"# Evaluation: {session.name}",
        "",
        f"{len(session.samples)} images {session.counts()}. Truth: none=A=CONTINUE, "
        "edge=B=SLOW, center=C=STOP (docs/labeling.md).",
    ]
    if subsets:
        groups = ", ".join(f"{n} ({len(f)} images)" for n, f in subsets.items())
        lines.append(f"Groups: {groups}; other = images in no group.")
    if session.excluded:
        lines.append("Excluded: " + ", ".join(session.excluded))
    lines.append("C→A = center (truth STOP) judged CONTINUE: the safety-critical error.")
    lines += [
        "",
        "| model | prompt | group | n | accuracy | recall A / B / C | C→A | errors "
        "| p50 ms | p95 ms |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for key, groups in results.items():
        model, prompt = key.split(" | ")
        for group, s in groups.items():
            rec = " / ".join("-" if v is None else f"{v:.2f}" for v in s["recall"].values())
            title = GROUP_TITLES.get(group, group)
            lines.append(
                f"| {model} | {prompt} | {title} | {s['n']} | {_fmt(s['accuracy'])} | {rec} | "
                f"{s['stop_as_continue']} | {s['teacher_errors']} | {s['latency_ms']['p50']} | "
                f"{s['latency_ms']['p95']} |"
            )
    for key, groups in results.items():
        for group, s in groups.items():
            title = GROUP_TITLES.get(group, group)
            lines += [
                "",
                f"### Confusion matrix: {key} | {title} (rows = truth, cols = prediction)",
            ]
            lines += [
                "",
                "| truth \\ pred | A CONTINUE | B SLOW | C STOP |",
                "|---|---:|---:|---:|",
            ]
            for name, row in zip(s["confusion"]["rows_truth_cols_pred"], s["confusion"]["matrix"],
                                 strict=True):  # fmt: skip
                letter = LETTERS[Choice[name]]
                lines.append(f"| {letter} {name} | " + " | ".join(str(v) for v in row) + " |")
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- {k}: {v}" for k, v in skipped.items()]
    if session.problems:
        lines += ["", "## Problems in metadata.csv / subsets", ""]
        lines += [f"- {p}" for p in session.problems]
    lines += ["", "## Models", "", "```json", json.dumps(info, indent=2), "```", ""]
    return "\n".join(lines)


def parse_list(value: str | None, default: list[str]) -> list[str]:
    return default if value is None else [v.strip() for v in value.split(",") if v.strip()]


def main(
    argv: list[str] | None = None,
    make_judge: Callable[[str, str], Judge] | None = None,
    repo_root: Path = REPO_ROOT,
    check_ignored=None,
    http_get=httpx.get,
    out=print,
    eval_cfg: dict[str, Any] | None = None,
) -> int:
    eval_cfg = eval_cfg or load_config("eval.yaml")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--models", help=f"default: {','.join(eval_cfg['default_models'])}")
    ap.add_argument("--prompts", help=f"default: {','.join(eval_cfg['default_prompts'])}")
    ap.add_argument("--tag", help="write to <out_root>/<session>/<tag>/ so runs do not overwrite")
    args = ap.parse_args(argv)

    models = parse_list(args.models, list(eval_cfg["default_models"]))
    prompts = parse_list(args.prompts, list(eval_cfg["default_prompts"]))
    unknown = [p for p in prompts if p not in eval_cfg["teacher_prompts"]]
    if unknown or not prompts or not models:
        out(
            f"error: unknown or empty --prompts {unknown}; "
            f"choose from {list(eval_cfg['teacher_prompts'])}"
        )
        return 2

    capture, teacher_cfg = load_config("capture.yaml"), load_config("teacher.yaml")
    choices_cfg = load_config("choices.yaml")
    session = load_session(repo_root / capture["out_root"], args.session, list(capture["labels"]))
    references: dict[str, np.ndarray] = {}
    for p in prompts:
        ref = ((eval_cfg.get("teacher_modes") or {}).get(p) or {}).get("reference")
        if ref:
            ref_path = session.root / ref
            if not ref_path.exists():
                out(f"error: reference image {ref} for prompt {p} not found in {session.name}")
                return 2
            references[p] = load_rgb(ref_path)
            if ref not in session.excluded:
                session.excluded.append(ref)
    session.samples = [s for s in session.samples if s.file not in session.excluded]
    subsets, subset_problems = resolve_subsets(
        (eval_cfg.get("subsets") or {}).get(session.name), session
    )
    session.problems += subset_problems
    out_dir = repo_root / eval_cfg["out_root"] / args.session
    if args.tag:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
            out(f"error: --tag {args.tag!r} may only contain letters, digits, . _ -")
            return 2
        out_dir = out_dir / args.tag
    check_ignored = check_ignored or (lambda p: is_git_ignored(p, repo_root))
    if not check_ignored(out_dir / "predictions.csv"):
        out(f"error: {out_dir.relative_to(repo_root)} is not git-ignored; refusing to write there")
        return 2
    info: dict[str, Any] = {}
    if make_judge is None:
        make_judge, info = real_factories(eval_cfg, teacher_cfg, choices_cfg, http_get, references)

    for p in session.problems:
        out(f"warning: {p}")
    out(f"session {session.name}: {len(session.samples)} images {session.counts()}")
    all_rows, results, skipped = [], {}, {}
    for model in models:
        for prompt in prompts:
            key = f"{model} | {prompt}"
            try:
                judge = make_judge(model, prompt)
            except Skip as e:
                skipped[key] = str(e)
                out(f"skip {key}: {e}")
                continue
            rows = evaluate(session, judge, model, prompt, eval_cfg["label_to_choice"],
                            int(eval_cfg["warmup"]), subsets)  # fmt: skip
            all_rows += rows
            results[key] = summarize_groups(rows, list(subsets))
            s = results[key]["all"]
            by_group = ", ".join(
                f"{GROUP_TITLES.get(g, g)} {_fmt(r['accuracy'])}" for g, r in results[key].items()
            )
            out(f"{key}: accuracy {by_group}; p50 {s['latency_ms']['p50']} ms, "
                f"p95 {s['latency_ms']['p95']} ms, errors {s['teacher_errors']}")  # fmt: skip

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    summary = {
        "session": session.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "images": len(session.samples),
        "counts": session.counts(),
        "label_to_choice": eval_cfg["label_to_choice"],
        "subsets": {n: sorted(f) for n, f in subsets.items()},
        "excluded": session.excluded,
        "warmup": int(eval_cfg["warmup"]),
        "results": results,
        "skipped": skipped,
        "problems": session.problems,
        "models": info,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "summary.md").write_text(summary_markdown(session, results, skipped, info, subsets))
    (out_dir / "prompts.md").write_text(
        prompts_markdown(prompts, eval_cfg, teacher_cfg, choices_cfg)
    )
    out(
        f"-> {out_dir.relative_to(repo_root)}/ "
        "(predictions.csv, summary.json, summary.md, prompts.md)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
