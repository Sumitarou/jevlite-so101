"""Fetch the SO-101 meshes (and, for maintainers, the URDF) from a pinned SO-ARM100 commit.

    python3 scripts/sim/fetch_so101_assets.py      # meshes -> so101_description/meshes/
    python3 scripts/sim/fetch_so101_assets.py --write-urdf --update-lock   # maintainers only

Source: https://github.com/TheRobotStudio/SO-ARM100 (Apache-2.0), commit pinned below.
The meshes (13 STL files, about 15 MB) are not committed; every file is checked against the
sha256 in assets.lock and the script stops on any mismatch. --write-urdf regenerates
urdf/so101_new_calib.urdf with the only change being the mesh paths
(assets/<name>.stl -> package://so101_description/meshes/<name>.stl).
Standard library only, so it runs with the system Python or the uv one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = "TheRobotStudio/SO-ARM100"
COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
UPSTREAM_DIR = "Simulation/SO101"
URDF_NAME = "so101_new_calib.urdf"
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "sim_ws" / "src" / "so101_description"
LOCK = PKG / "assets.lock"
MESH_PREFIX = "package://so101_description/meshes/"


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"


def download(path: str) -> bytes:
    with urllib.request.urlopen(raw_url(path), timeout=60) as r:
        return r.read()


def rewrite_mesh_paths(urdf: str) -> tuple[str, list[str]]:
    """assets/<name>.stl -> package://so101_description/meshes/<name>.stl; returns the names."""
    names: list[str] = []

    def sub(m: re.Match) -> str:
        names.append(m.group(1))
        return f'filename="{MESH_PREFIX}{m.group(1)}"'

    out = re.sub(r'filename="assets/([^"/]+\.stl)"', sub, urdf)
    if 'filename="assets/' in out:
        raise ValueError("unexpected mesh path left in the URDF")
    return out, sorted(set(names))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_lock() -> dict:
    return json.loads(LOCK.read_text())


def fetch_meshes(dest: Path) -> int:
    lock = load_lock()
    if lock["commit"] != COMMIT:
        raise SystemExit(f"assets.lock is for commit {lock['commit']}, script pins {COMMIT}")
    dest.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for name, digest in lock["meshes"].items():
        target = dest / name
        if target.exists() and sha256(target.read_bytes()) == digest:
            continue
        data = download(f"{UPSTREAM_DIR}/assets/{name}")
        if sha256(data) != digest:
            raise SystemExit(f"sha256 mismatch for {name}; refusing to write it")
        target.write_bytes(data)
        fetched += 1
    print(f"meshes: {len(lock['meshes'])} ok ({fetched} downloaded) in {dest.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-urdf", action="store_true", help="regenerate urdf/ from upstream")
    ap.add_argument("--update-lock", action="store_true", help="rewrite assets.lock (maintainers)")
    args = ap.parse_args(argv)

    if args.write_urdf or args.update_lock:
        urdf, names = rewrite_mesh_paths(download(f"{UPSTREAM_DIR}/{URDF_NAME}").decode())
        if args.write_urdf:
            (PKG / "urdf" / URDF_NAME).write_text(urdf)
            (PKG / "LICENSE").write_bytes(download("LICENSE"))
            print(f"wrote urdf/{URDF_NAME} ({len(names)} meshes referenced) and LICENSE")
        if args.update_lock:
            meshes = {n: sha256(download(f"{UPSTREAM_DIR}/assets/{n}")) for n in names}
            lock = {
                "repo": REPO,
                "commit": COMMIT,
                "path": f"{UPSTREAM_DIR}/assets",
                "meshes": meshes,
            }
            LOCK.write_text(json.dumps(lock, indent=2) + "\n")
            print(f"wrote assets.lock ({len(meshes)} meshes)")
    return fetch_meshes(PKG / "meshes")


if __name__ == "__main__":
    sys.exit(main())
