"""Judge every image under a folder with the Ollama teacher (Step 0: print only).

    uv run python scripts/label_teacher.py --dir samples/ [--model qwen2.5vl:7b] [--csv out.csv]

Full labelling (saving labels, comparing with humans) comes in a later step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jevlite.teacher_ollama import OllamaTeacher
from scripts.run_zeroshot_on_images import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("samples"))
    ap.add_argument("--model", default=None, help="default: default_model in configs/teacher.yaml")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args(argv)

    teacher = OllamaTeacher(model=args.model)
    print(f"teacher: {teacher.model}")
    stats = run(teacher, args.dir, args.csv)
    if teacher.last_error:
        print(f"last error: {teacher.last_error}", file=sys.stderr)
    return 0 if stats["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
