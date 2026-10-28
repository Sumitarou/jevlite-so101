"""Read one demo episode (rosbag2 mcap) without ROS, using rosbags. Times are header stamps (ns)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sim.bag_align import reorder, stamp_ns  # noqa: E402


@dataclass
class Episode:
    bag: Path
    info: dict  # episode.json
    t_action: np.ndarray  # (M,) ns
    action: np.ndarray  # (M, J) in `joints` order
    t_state: np.ndarray  # (N,) ns
    state: np.ndarray  # (N, J) in `joints` order
    t_image: np.ndarray  # (K,) ns
    images: list[np.ndarray] = field(repr=False)  # K x (H, W, 3) uint8 RGB


def episode_dirs(src: Path) -> list[Path]:
    return sorted(p for p in src.iterdir() if p.is_dir() and (p / "episode.json").exists())


def read_episode(bag: Path, topics: dict, joints: list[str], with_images: bool = True) -> Episode:
    t_a, a, t_s, s, t_i, imgs = [], [], [], [], [], []
    wanted = {topics["action"], topics["state"]} | ({topics["image"]} if with_images else set())
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as r:
        conns = [c for c in r.connections if c.topic in wanted]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            t = stamp_ns(m.header.stamp.sec, m.header.stamp.nanosec)
            if c.topic == topics["action"]:
                t_a.append(t)
                a.append(np.asarray(m.position, np.float64)[reorder(list(m.name), joints)])
            elif c.topic == topics["state"]:
                t_s.append(t)
                s.append(np.asarray(m.position, np.float64)[reorder(list(m.name), joints)])
            else:
                if m.encoding != "rgb8":
                    raise ValueError(f"{bag.name}: image encoding {m.encoding} (need rgb8)")
                t_i.append(t)
                imgs.append(
                    np.frombuffer(bytes(m.data), np.uint8)
                    .reshape(m.height, m.step)[:, : m.width * 3]
                    .reshape(m.height, m.width, 3)
                    .copy()
                )

    t_a, a = np.asarray(t_a, np.int64), np.asarray(a)
    t_s, s = np.asarray(t_s, np.int64), np.asarray(s)
    for name, t in (("action", t_a), ("state", t_s)):
        if np.any(np.diff(t) < 0):
            raise ValueError(f"{bag.name}: {name} stamps not in order")
    t_i = np.asarray(t_i, np.int64)
    if with_images and np.any(np.diff(t_i) < 0):
        order = np.argsort(t_i, kind="stable")
        t_i, imgs = t_i[order], [imgs[k] for k in order]
    info = json.loads((bag / "episode.json").read_text())
    return Episode(bag, info, t_a, a, t_s, s, t_i, imgs)
