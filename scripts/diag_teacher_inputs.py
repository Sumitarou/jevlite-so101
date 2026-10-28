"""Diagnose whether images actually reach the Ollama teachers (not committed; checks only).

    uv run python scripts/diag_teacher_inputs.py --session 20261011-desk

1. prompt_eval_count with / without the image (same request as the evaluation otherwise)
2. the image as sent: size, format, bytes, channel order compared with the source file
3. free-form Japanese description (no choices, no JSON format) for 3 images
4. position bias: the same question with the choice letters permuted
Prints a report and writes it as JSON next to the evaluation outputs (git-ignored).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys

import httpx
import numpy as np
from PIL import Image

from jevlite.config import load_config
from scripts.capture_samples import REPO_ROOT
from scripts.eval_samples import teacher_prompt

MODELS = ["qwen2.5vl:7b", "gemma3:4b"]
FREE_PROMPT = "この画像に何が写っているか、白い面と緑の帯に触れて日本語で2〜3文で説明してください。"
FREE_FILES = ["none/00001.jpg", "edge/00001.jpg", "center/00015.jpg"]
# meaning -> definition used in the letter prompts (same content as def-v1)
MEANINGS = {
    "NONE": "there is no hand and no object on the white surface or on the green band",
    "SLOW": "an object or hand is on the GREEN band only and does not reach the white surface",
    "STOP": "any object or hand overlaps the WHITE surface, even partially",
}
ORDERS = [["NONE", "SLOW", "STOP"], ["STOP", "NONE", "SLOW"], ["SLOW", "STOP", "NONE"]]


def letter_prompt(order: list[str]) -> str:
    lines = [
        "You are a safety monitor for a small desktop robot arm. The camera looks down at the",
        "work table. The table has a WHITE surface (the protected area) surrounded by a GREEN",
        "tape band about 5 cm wide (the warning area). Answer with exactly one letter:",
    ]
    lines += [f"- {letter}: {MEANINGS[m]}." for letter, m in zip("ABC", order, strict=True)]
    lines.append("Respond in JSON only.")
    return "\n".join(lines)


def chat(client: httpx.Client, url: str, body: dict) -> dict:
    r = client.post(f"{url}/api/chat", json=body)
    r.raise_for_status()
    return r.json()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    from jevlite.teacher_ollama import OllamaTeacher, encode_image

    teacher_cfg, eval_cfg = load_config("teacher.yaml"), load_config("eval.yaml")
    capture = load_config("capture.yaml")
    sdir = REPO_ROOT / capture["out_root"] / args.session
    url = teacher_cfg["ollama_url"].rstrip("/")
    client = httpx.Client(timeout=float(teacher_cfg.get("timeout_s", 120)))
    report: dict = {"session": args.session}

    def load(f: str) -> np.ndarray:
        with Image.open(sdir / f) as im:
            return np.asarray(im.convert("RGB"))

    probe = "center/00015.jpg"
    img = load(probe)

    # 2. the image as sent
    b64 = encode_image(img, teacher_cfg.get("image_format", "jpeg"),
                       int(teacher_cfg.get("jpeg_quality", 90)))  # fmt: skip
    raw = base64.b64decode(b64)
    sent = Image.open(io.BytesIO(raw))
    sent_arr = np.asarray(sent.convert("RGB")).astype(float)
    src_mean = img.reshape(-1, 3).mean(0)
    sent_mean = sent_arr.reshape(-1, 3).mean(0)
    report["image_sent"] = {
        "file": probe,
        "source_size": [img.shape[1], img.shape[0]],
        "sent_size": list(sent.size),
        "sent_format": sent.format,
        "sent_mode": sent.mode,
        "sent_bytes": len(raw),
        "base64_chars": len(b64),
        "source_mean_rgb": [round(v, 1) for v in src_mean],
        "sent_mean_rgb": [round(v, 1) for v in sent_mean],
        "mean_abs_diff_same_order": round(float(np.abs(sent_mean - src_mean).mean()), 2),
        "mean_abs_diff_if_bgr": round(float(np.abs(sent_mean[::-1] - src_mean).mean()), 2),
        "pixel_std": round(float(sent_arr.std()), 1),
    }

    # 1. prompt_eval_count with / without image (both orders, to see prompt-cache effects)
    report["prompt_eval_count"] = []
    for model in MODELS:
        for pname in ["baseline", "def-v1"]:
            cfg = {**teacher_cfg, "prompt": teacher_prompt(pname, eval_cfg, teacher_cfg)}
            body = OllamaTeacher(cfg, model=model).build_request(img)
            no_img = json.loads(json.dumps(body))
            no_img["messages"][0].pop("images")
            for order in (["no_image", "image"], ["image", "no_image"]):
                counts = {}
                for kind in order:
                    res = chat(client, url, body if kind == "image" else no_img)
                    counts[kind] = {
                        "prompt_eval_count": res.get("prompt_eval_count"),
                        "answer": res["message"]["content"],
                    }
                report["prompt_eval_count"].append(
                    {"model": model, "prompt": pname, "order": order, **counts}
                )

    # 3. free-form description
    report["free_text"] = []
    for model in MODELS:
        for f in FREE_FILES:
            b = base64.b64encode(raw if f == probe else base64.b64decode(
                encode_image(load(f), "jpeg", 90))).decode()  # fmt: skip
            res = chat(client, url, {
                "model": model, "stream": False, "options": {"temperature": 0},
                "messages": [{"role": "user", "content": FREE_PROMPT, "images": [b]}],
            })  # fmt: skip
            report["free_text"].append(
                {
                    "model": model,
                    "file": f,
                    "text": res["message"]["content"].strip(),
                    "prompt_eval_count": res.get("prompt_eval_count"),
                }  # fmt: skip
            )

    # 4. position bias on center/00015 (truth: STOP)
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": list("ABC")}},
              "required": ["choice"]}  # fmt: skip
    report["letters"] = []
    for model in MODELS:
        for order in ORDERS:
            for with_image in (True, False):
                msg = {"role": "user", "content": letter_prompt(order)}
                if with_image:
                    msg["images"] = [b64]
                body = {"model": model, "stream": False, "format": schema,
                        "options": {"temperature": 0}, "messages": [msg]}  # fmt: skip
                res = chat(client, url, body)
                try:
                    letter = json.loads(res["message"]["content"])["choice"]
                except Exception:
                    letter = res["message"]["content"]
                meaning = dict(zip("ABC", order, strict=True)).get(letter, "?")
                report["letters"].append(
                    {
                        "model": model,
                        "order": "".join(o[0] for o in order) + f" ({'/'.join(order)})",
                        "image": with_image,
                        "letter": letter,
                        "meaning": meaning,
                    }  # fmt: skip
                )

    out = REPO_ROOT / eval_cfg["out_root"] / args.session / "diag_teacher_inputs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
