"""M4b: the image judges' common parts (jevlite/student.py) and the metrics (judge_data.py)."""

import csv

import numpy as np
import pytest
import yaml

from jevlite.api import Choice, Judge
from jevlite.student import (
    ZONES,
    LinearHead,
    ZoneJudge,
    bias_vector,
    check_box,
    choice_probs,
    softmax_rows,
    zone_choices,
)
from scripts.sim.fit_camera import project
from scripts.sim.intruder import load_m4a
from scripts.sim.intruder_sampling import load_judge_sim
from scripts.sim.judge_data import better, load_rows, rows_digest, subset_masks, zone_metrics
from scripts.sim.kinematics import SIM_YAML

SIM = yaml.safe_load(SIM_YAML.read_text())
CFG = load_judge_sim()
ZONE_MAP = load_m4a()["zones"]


def test_crop_box_holds_the_band_with_a_margin():
    c, t = SIM["camera"], SIM["table"]
    hx, hy = t["white"][0] / 2 + t["band_width"], t["white"][1] / 2 + t["band_width"]
    pts = np.array([[sx * hx, sy * hy, 0.002] for sx in (-1, 1) for sy in (-1, 1)])
    u, v, _ = project(pts, c["xyz"][0], c["xyz"][2], c["rpy"][1], c["width"], c["height"],
                      c["hfov"])  # fmt: skip
    u0, v0, u1, v1 = check_box(CFG["crop"], c["width"], c["height"])
    assert u0 + 20 <= u.min() and u.max() <= u1 - 20
    assert v0 + 50 <= v.min() and v.max() <= v1


def test_check_box_rejects_boxes_outside_the_image():
    with pytest.raises(ValueError):
        check_box([0, 0, 641, 480], 640, 480)
    with pytest.raises(ValueError):
        check_box([10, 10, 10, 20], 640, 480)


def test_zone_choices_follow_the_m4a_map():
    assert zone_choices(ZONE_MAP) == [Choice.CONTINUE, Choice.SLOW, Choice.STOP]
    with pytest.raises(ValueError):
        zone_choices({"none": "CONTINUE", "edge": "SLOW"})


def test_choice_probs_sum_shared_choices():
    p = choice_probs(np.array([0.2, 0.3, 0.5]), [Choice.CONTINUE, Choice.STOP, Choice.STOP])
    assert p == pytest.approx({"CONTINUE": 0.2, "SLOW": 0.0, "STOP": 0.8})


def test_softmax_rows():
    p = softmax_rows([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    assert p[0] == pytest.approx([1 / 3] * 3)
    assert p[1] == pytest.approx([1, 0, 0])


def test_bias_vector():
    assert bias_vector(None).tolist() == [0, 0, 0]
    assert bias_vector({"center": 1.5}).tolist() == [0, 0, 1.5]
    with pytest.raises(ValueError):
        bias_vector({"centre": 1.0})
    with pytest.raises(ValueError):
        bias_vector({"edge": float("nan")})


def test_zone_judge_decision():
    j = ZoneJudge(lambda im: np.array([[0.0, 1.0, 3.0]]), ZONE_MAP, "fake")
    assert isinstance(j, Judge)
    d = j.judge(np.zeros((480, 640, 3), np.uint8), t_capture=12.5)
    assert d.choice is Choice.STOP and d.source == "fake"
    assert d.t_mono == 12.5 and d.latency_ms >= 0
    assert sum(d.probs.values()) == pytest.approx(1.0)
    assert d.probs["STOP"] == pytest.approx(softmax_rows([[0, 1, 3]])[0][2])


def test_logit_bias_moves_toward_the_safe_side():
    logits = lambda im: np.array([[1.0, 0.0, 0.9]])  # noqa: E731
    plain = ZoneJudge(logits, ZONE_MAP, "fake")
    biased = ZoneJudge(logits, ZONE_MAP, "fake", {"center": 0.5})
    im = np.zeros((4, 4, 3), np.uint8)
    assert plain.judge(im).choice is Choice.CONTINUE
    assert biased.judge(im).choice is Choice.STOP


def test_linear_head_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    h = LinearHead(rng.normal(size=5), rng.uniform(1, 2, 5), rng.normal(size=(3, 5)),
                   rng.normal(size=3))  # fmt: skip
    x = rng.normal(size=(4, 5))
    h.save(tmp_path / "head.npz")
    h2 = LinearHead.load(tmp_path / "head.npz")
    assert np.allclose(h.logits(x), h2.logits(x))
    assert np.allclose(h.logits(x), ((x - h.mean) / h.scale) @ h.coef.T + h.intercept)
    with pytest.raises(ValueError):
        LinearHead(np.zeros(5), np.ones(5), np.zeros((2, 5)), np.zeros(3))


def test_zone_metrics():
    y = np.array([0, 0, 1, 1, 2, 2, 2])
    p = np.array([0, 1, 1, 0, 2, 0, 1])
    m = zone_metrics(y, p)
    assert m["n"] == 7 and m["accuracy"] == pytest.approx(3 / 7, abs=1e-4)
    assert m["confusion"] == [[1, 1, 0], [1, 1, 0], [1, 1, 1]]
    assert m["center_to_none"] == 1 and m["center_to_edge"] == 1 and m["edge_to_none"] == 1
    assert m["under"] == 3 and m["over"] == 1
    assert m["recall"]["center"] == pytest.approx(1 / 3, abs=1e-4)


def test_better_prefers_accuracy_then_fewer_center_to_none():
    a = {"accuracy": 0.9, "center_to_none": 2}
    assert better(a, None)
    assert better(a, {"accuracy": 0.8, "center_to_none": 0})
    assert better({"accuracy": 0.9, "center_to_none": 1}, a)
    assert not better(a, a)


def _write_data(d, rows):
    (d / "images").mkdir(parents=True)
    with (d / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_load_rows_subsets_and_digest(tmp_path):
    base = {"kind": "table", "arm_near": "0", "d_boundary": "0.05"}
    rows = [{**base, "file": "a.png", "zone": "none", "split": "train"},
            {**base, "file": "b.png", "zone": "center", "split": "val", "d_boundary": "0.004"},
            {**base, "file": "c.png", "zone": "edge", "split": "test", "arm_near": "1",
             "kind": "park"}]  # fmt: skip
    _write_data(tmp_path / "run", rows)
    got = load_rows([tmp_path / "run"])
    assert [r["path"].endswith(f"run/images/{r['file']}") for r in got] == [True] * 3
    m = subset_masks(got, near_m=CFG["data"]["near_eval_m"])
    assert m["near"].tolist() == [False, True, False]
    assert m["arm_near"].tolist() == [False, False, True]
    assert m["on_table"].tolist() == [True, True, False]
    assert rows_digest(got) == rows_digest(load_rows([tmp_path / "run"]))
    _write_data(tmp_path / "bad", [{**rows[0], "zone": "middle"}])
    with pytest.raises(ValueError, match="bad row"):
        load_rows([tmp_path / "bad"])


def test_config_zones_order_matches():
    assert ZONES == ("none", "edge", "center")
    for kind in ("v1", "v2"):
        assert set(CFG["logit_bias"][kind]) == set(ZONES)
        assert all(v >= 0 for v in CFG["logit_bias"][kind].values())  # only toward the safe side
        assert CFG["logit_bias"][kind]["none"] == 0.0
    assert set(CFG["v0"]["prompts"]) == set(ZONES)


@pytest.mark.gpu
def test_siglip_features_shapes():
    from jevlite.student import SiglipFeatures

    fx = SiglipFeatures(CFG["siglip"], CFG["crop"], ["pooled", "grid2"])
    f = fx([np.zeros((480, 640, 3), np.uint8)] * 2)
    assert f.shape == (2, 768 * 5)
