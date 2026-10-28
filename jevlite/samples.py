"""Read a capture session (samples/private/<session>/) written by scripts/capture_samples.py.

The ground truth is the `label` column of metadata.csv. Each row is checked against the file
on disk: a missing file is skipped, and a file whose folder differs from its metadata label is
kept (metadata wins) but reported, so relabelling by moving files is not silently ignored.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    file: str  # relative to the session dir, e.g. "edge/00003.jpg"
    label: str  # none / edge / center
    teacher_choice: str  # A / B / C
    path: Path


@dataclass
class Session:
    name: str
    root: Path
    samples: list[Sample]
    problems: list[str]
    excluded: list[str] = field(default_factory=list)  # files left out (e.g. reference images)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.samples:
            out[s.label] = out.get(s.label, 0) + 1
        return out


def load_session(samples_root: Path, name: str, labels: list[str]) -> Session:
    root = samples_root / name
    meta = root / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found (is the session name right?)")
    samples, problems = [], []
    with meta.open(newline="") as f:
        for row in csv.DictReader(f):
            file, label = row["file"], row["label"]
            if label not in labels:
                problems.append(f"{file}: unknown label {label!r}")
                continue
            path = root / file
            if not path.exists():
                problems.append(f"{file}: listed in metadata.csv but missing")
                continue
            folder = Path(file).parts[0]
            if folder != label:
                problems.append(f"{file}: folder {folder!r} but metadata label {label!r}")
            samples.append(Sample(file, label, row.get("teacher_choice", ""), path))
    order = {lab: i for i, lab in enumerate(labels)}
    samples.sort(key=lambda s: (order[s.label], s.file))
    return Session(name, root, samples, problems)


def resolve_subsets(
    spec: dict[str, list[str]] | None, session: Session
) -> tuple[dict[str, set[str]], list[str]]:
    """Expand {"boundary": ["edge/00001-00009", "center/00004"]} into sets of session files.

    Returns (subset -> files, problems). Entries that match no sample are reported, not ignored.
    """
    files = {s.file for s in session.samples}
    out: dict[str, set[str]] = {}
    problems: list[str] = []
    for name, entries in (spec or {}).items():
        if name in ("all", "other"):
            raise ValueError(f"subset name {name!r} is reserved")
        chosen: set[str] = set()
        for entry in entries:
            m = re.fullmatch(r"([^/]+)/(\d+)(?:-(\d+))?", entry)
            if not m:
                raise ValueError(
                    f"bad subset entry {entry!r} (use label/00001 or label/00001-00009)"
                )
            label, start = m[1], m[2]
            end = m[3] or start
            for n in range(int(start), int(end) + 1):
                f = f"{label}/{n:0{len(start)}d}.jpg"
                if f in files:
                    chosen.add(f)
                else:
                    problems.append(f"subset {name}: {f} is not in session {session.name}")
        out[name] = chosen
    return out, problems
