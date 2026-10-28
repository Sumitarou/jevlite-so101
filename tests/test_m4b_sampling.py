"""M4b: rod placements for the judge's training images (scripts/sim/intruder_sampling.py)."""

import math
from collections import Counter

import numpy as np
import pytest
import yaml

from scripts.sim.intruder import ZONES, load_m4a, zone_of
from scripts.sim.intruder_sampling import (
    IntruderSampler,
    boundary_distance,
    foot_in_view,
    footprint_distances,
    load_judge_sim,
    rect_signed,
    split_of,
    zone_from_distances,
)
from scripts.sim.kinematics import SIM_YAML

SIM = yaml.safe_load(SIM_YAML.read_text())
M4A = load_m4a()
DATA = load_judge_sim()["data"]
R = SIM["intruder"]["radius"]
WX, WY = SIM["table"]["white"][0] / 2, SIM["table"]["white"][1] / 2
B = SIM["table"]["band_width"]


def test_rect_signed():
    assert rect_signed(1, 1, 0, 0) == -1
    assert rect_signed(1, 1, 2, 0) == 1
    assert rect_signed(1, 1, 2, 2) == pytest.approx(math.sqrt(2))
    assert rect_signed(1, 2, 0.5, 0) == -0.5


def test_zone_from_distances_matches_zone_of_on_a_grid():
    rng = np.random.default_rng(0)
    for x, y in rng.uniform(-0.3, 0.3, size=(3000, 2)):
        assert zone_from_distances(*footprint_distances(x, y, SIM)) == zone_of(x, y, SIM)


@pytest.mark.parametrize("x,y,zone", [
    (0.0, WY + R, "center"),            # touching the white area
    (0.0, WY + R + 1e-4, "edge"),
    (0.0, WY + B + R, "edge"),          # touching the band's outer edge
    (0.0, WY + B + R + 1e-4, "none"),
])  # fmt: skip
def test_boundaries(x, y, zone):
    assert zone_from_distances(*footprint_distances(x, y, SIM)) == zone
    assert boundary_distance(x, y, SIM) < 2e-4


def test_m4a_places_are_in_view_and_park_is_not():
    p = M4A["path"]
    for name in ("edge", "center", "boundary"):
        assert foot_in_view(p["x"], p["places"][name], SIM, M4A)
    assert not foot_in_view(p["x"], p["places"]["out"], SIM, M4A)
    park = SIM["intruder"]["park"]
    assert not foot_in_view(park[0], park[1], SIM, M4A)


def _samples(n, forbidden=None, seed=0):
    s = IntruderSampler(DATA, SIM, M4A)
    rng = np.random.default_rng(seed)
    return [s.sample(rng, forbidden) for _ in range(n)]


def test_sampled_zones_kinds_and_near_fraction():
    out = _samples(3000)
    s = DATA["sampling"]
    kinds = Counter(o["kind"] for o in out)
    assert kinds["park"] / 3000 == pytest.approx(s["park_frac"], abs=0.025)
    assert kinds["path"] / 3000 == pytest.approx(s["path_frac"], abs=0.03)
    placed = [o for o in out if o["kind"] != "park"]
    for o in placed:
        assert zone_of(o["x"], o["y"], SIM) == o["target"]
        assert foot_in_view(o["x"], o["y"], SIM, M4A)
        if o["near"]:
            assert boundary_distance(o["x"], o["y"], SIM) <= s["near_m"] + 1e-12
        if o["kind"] == "path":
            assert abs(o["x"] - M4A["path"]["x"]) <= s["path_dx"] + 1e-12
            assert o["y"] >= 0.0
    zones = Counter(o["target"] for o in placed)
    for z in ZONES:  # 1 : 1 : 1
        assert zones[z] / len(placed) == pytest.approx(1 / 3, abs=0.04)
    near = sum(o["near"] for o in placed) / len(placed)
    assert near == pytest.approx(s["near_frac"], abs=0.04)


def test_rod_height_is_the_m4a_hover():
    o = next(o for o in _samples(50) if o["kind"] != "park")
    assert o["z"] == pytest.approx(M4A["path"]["hover"] + SIM["intruder"]["length"] / 2)


def test_forbidden_places_are_avoided():
    def forbidden(x, y):
        return math.hypot(x, y + 0.04) < 0.06

    for o in _samples(500, forbidden):
        if o["kind"] != "park":
            assert math.hypot(o["x"], o["y"] + 0.04) >= 0.06


def test_impossible_request_raises():
    with pytest.raises(RuntimeError, match="no .* placement"):
        _samples(50, forbidden=lambda x, y: True)


def test_same_seed_same_samples():
    assert _samples(20, seed=3) == _samples(20, seed=3)


def test_bad_weights_are_rejected():
    bad = {**DATA, "sampling": {**DATA["sampling"], "zones": {"none": 1, "edge": 1}}}
    with pytest.raises(ValueError, match="zones"):
        IntruderSampler(bad, SIM, M4A)
    bad = {**DATA, "sampling": {**DATA["sampling"], "park_frac": 0.6, "path_frac": 0.6}}
    with pytest.raises(ValueError, match="park_frac"):
        IntruderSampler(bad, SIM, M4A)


def test_split_is_per_seed_and_roughly_70_15_15():
    sp = DATA["split"]
    a = [split_of(s, sp) for s in range(2000, 4000)]
    assert a == [split_of(s, sp) for s in range(2000, 4000)]
    c = Counter(a)
    assert c["train"] / 2000 == pytest.approx(sp["train"], abs=0.03)
    assert c["val"] / 2000 == pytest.approx(sp["val"], abs=0.03)
    assert set(c) == {"train", "val", "test"}
