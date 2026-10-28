"""Make one overview image of a capture session to check the labels by eye.

    uv run python scripts/contact_sheet.py --session 20261011-desk

Every image is shrunk, captioned with "<choice> <label> <number>" on a bar in the label's
colour, grouped by label (none, edge, center), and saved to outputs/contact_<session>.jpg
(git-ignored). Labels come from metadata.csv; see docs/labeling.md for their definitions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from jevlite.config import load_config
from jevlite.samples import Session, load_session, resolve_subsets
from scripts.capture_samples import REPO_ROOT, is_git_ignored

CAPTION_H = 22
HEADER_H = 30
GAP = 4


def make_sheet(
    session: Session, labels: list[str], cfg: dict, subsets: dict[str, set[str]] | None = None
) -> Image.Image:
    subsets = subsets or {}
    tw, cols, colors = int(cfg["thumb_width"]), int(cfg["columns"]), cfg["colors"]
    font = ImageFont.load_default()
    thumbs: dict[str, list[Image.Image]] = {lab: [] for lab in labels}
    th = None
    for s in session.samples:
        with Image.open(s.path) as im:
            im = im.convert("RGB")
            th = th or round(im.height * tw / im.width)
            thumb = im.resize((tw, th))
        cell = Image.new("RGB", (tw, th + CAPTION_H), colors[s.label])
        cell.paste(thumb, (0, 0))
        number = Path(s.file).stem
        tags = " ".join(f"[{n}]" for n, files in subsets.items() if s.file in files)
        caption = f"{s.teacher_choice} {s.label} {number} {tags}".rstrip()
        ImageDraw.Draw(cell).text((4, th + 4), caption, fill="white", font=font)
        thumbs[s.label].append(cell)
    th = th or tw * 3 // 4
    cell_h = th + CAPTION_H
    width = cols * (tw + GAP) + GAP
    height = HEADER_H
    for lab in labels:
        rows = max(1, -(-len(thumbs[lab]) // cols))
        height += HEADER_H + rows * (cell_h + GAP)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    counts = session.counts()
    title = f"session {session.name}: " + ", ".join(f"{lab} {counts.get(lab, 0)}" for lab in labels)
    draw.text((GAP, 8), title, fill="black", font=font)
    y = HEADER_H
    for lab in labels:
        draw.rectangle([0, y, width, y + HEADER_H - 4], fill=colors[lab])
        draw.text((GAP, y + 8), f"{lab} ({len(thumbs[lab])})", fill="white", font=font)
        y += HEADER_H
        for i, cell in enumerate(thumbs[lab]):
            r, c = divmod(i, cols)
            sheet.paste(cell, (GAP + c * (tw + GAP), y + r * (cell_h + GAP)))
        y += max(1, -(-len(thumbs[lab]) // cols)) * (cell_h + GAP)
    return sheet


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT, check_ignored=None, out=print):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)

    capture = load_config("capture.yaml")
    cfg = load_config("eval.yaml")["contact_sheet"]
    labels = list(capture["labels"])
    session = load_session(repo_root / capture["out_root"], args.session, labels)
    spec = (load_config("eval.yaml").get("subsets") or {}).get(args.session)
    subsets, subset_problems = resolve_subsets(spec, session)
    session.problems += subset_problems
    out_path = repo_root / cfg["out_dir"] / f"contact_{args.session}.jpg"
    check_ignored = check_ignored or (lambda p: is_git_ignored(p, repo_root))
    if not check_ignored(out_path):
        out(f"error: {out_path.relative_to(repo_root)} is not git-ignored; refusing to write it")
        return 2
    sheet = make_sheet(session, labels, cfg, subsets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=int(cfg["jpeg_quality"]))
    for p in session.problems:
        out(f"warning: {p}")
    marked = ", ".join(f"{n} {len(f)}" for n, f in subsets.items())
    out(f"{len(session.samples)} images {session.counts()}"
        + (f", marked: {marked}" if marked else "")
        + f" -> {out_path.relative_to(repo_root)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
