"""scripts/video/video_core.py: config checks and the timeline of the submission video."""

import copy
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from scripts.video import video_core as vc

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"paused": "一時停止", "normal": "×1（等倍）", "fast": "×{s} 倍速", "slow": "×{s} スロー"}


def tiny_cfg():
    return {
        "output": {"width": 1280, "height": 720, "fps": 30},
        "duration_limits_s": [1, 100],
        "sources": {"a": {"run": "r", "bag": "b"}},
        "sections": [
            {"name": "t", "kind": "title", "duration_s": 2},
            {"name": "c", "kind": "clip", "source": "a", "pre_hold_s": 1, "hold_s": 1,
             "segments": [{"from": 0, "to": 2, "speed": 1}, {"from": 2, "to": 6, "speed": 2}],
             "captions": [{"at": 0, "until": 6, "text": "x"}]},
        ],
    }  # fmt: skip


def test_timeline_mixes_rates_and_holds():
    tl = vc.build_timeline([{"from": 0, "to": 2, "speed": 1}, {"from": 2, "to": 6, "speed": 2}],
                           fps=30, pre_hold_s=1, hold_s=0.5)  # fmt: skip
    assert len(tl) == 30 + 60 + 60 + 15
    assert tl[0] == (0.0, 0.0) and tl[29] == (0.0, 0.0)  # paused before
    assert tl[30] == (0.0, 1.0) and tl[31][0] == pytest.approx(1 / 30)
    assert tl[90] == (2.0, 2.0) and tl[91][0] == pytest.approx(2 + 2 / 30)
    assert tl[149][0] == pytest.approx(6 - 2 / 30)
    assert tl[-1] == (6.0, 0.0)
    assert all(b[0] >= a[0] for a, b in zip(tl, tl[1:], strict=False))  # sim time never goes back


def test_slow_motion_has_more_frames_than_sim_time():
    tl = vc.build_timeline([{"from": 4, "to": 5, "speed": 0.5}], fps=30)
    assert len(tl) == 60 and tl[1][0] == pytest.approx(4 + 0.5 / 30)


def test_speed_label_names_every_rate_but_one():
    assert vc.speed_label(1.0, LABELS) == "×1（等倍）"
    assert vc.speed_label(2.0, LABELS) == "×2 倍速"
    assert vc.speed_label(1.5, LABELS) == "×1.5 倍速"
    assert vc.speed_label(0.5, LABELS) == "×0.5 スロー"
    assert vc.speed_label(0.0, LABELS) == "一時停止"


def test_frame_and_tick_lookup():
    st = [10.0, 10.1, 10.2]
    assert vc.frame_index(st, 9.0) == 0
    assert vc.frame_index(st, 10.1) == 1
    assert vc.frame_index(st, 10.15) == 1
    assert vc.frame_index(st, 99) == 2
    assert vc.last_at(st, 9.0) == -1


def test_captions_and_wrap():
    caps = [{"at": 0, "until": 2, "text": "a"}, {"at": 2, "until": 4, "text": "b"}]
    assert vc.active_captions(caps, 1.99) == ["a"] and vc.active_captions(caps, 2.0) == ["b"]
    assert vc.active_captions(caps, 4.0) == []
    lines = vc.wrap_text("あいうえおかき\nく", 3, len)
    assert lines == ["あいう", "えおか", "き", "く"]
    # ASCII words and numbers move to the next line whole
    assert vc.wrap_text("減速（SLOW）", 6, len) == ["減速", "（SLOW）"]  # no "（" at a line end
    assert vc.wrap_text("あいう。え", 3, len) == ["あいう。", "え"]  # "。" hangs, never starts
    assert vc.wrap_text("x ±2.5 cm", 5, len) == ["x", "±2.5", "cm"]


def test_validate_accepts_and_counts():
    cfg = tiny_cfg()
    vc.validate(cfg)
    assert vc.total_duration_s(cfg) == pytest.approx(2 + 1 + 2 + 2 + 1)


@pytest.mark.parametrize("breakit, msg", [
    (lambda c: c["sections"][1]["segments"][1].update(speed=0), "speed"),
    (lambda c: c["sections"][1]["segments"][1].update({"from": 1}), "back in time"),
    (lambda c: c["sections"][1]["segments"][0].update(to=0), "after from"),
    (lambda c: c["sections"][1].update(source="zzz"), "unknown source"),
    (lambda c: c["sections"][0].update(kind="movie"), "unknown kind"),
    (lambda c: c["sections"][1].update(name="t"), "duplicated"),
    (lambda c: c["sections"][1]["captions"][0].update(until=60), "after the section"),
    (lambda c: c.update(duration_limits_s=[60, 180]), "outside"),
])  # fmt: skip
def test_validate_rejects(breakit, msg):
    cfg = copy.deepcopy(tiny_cfg())
    breakit(cfg)
    with pytest.raises(ValueError, match=msg):
        vc.validate(cfg)


def test_repo_config_is_valid_and_within_the_submission_limits():
    cfg = yaml.safe_load((ROOT / "configs" / "video.yaml").read_text())
    vc.validate(cfg)
    total = vc.total_duration_s(cfg)
    assert 60 <= total <= 180
    assert abs(total - cfg["target_s"]) <= 15
    assert cfg["output"]["width"] == 1280 and cfg["output"]["height"] == 720
    assert cfg["output"]["fps"] == 30
    # the gate sections say what the judge is: the image judge v2, or the oracle where a section
    # still shows it (with "シミュレータの正解"); a bag of an oracle run must not be called v2
    for sec in cfg["sections"]:
        if sec["kind"] in ("clip", "compare") and (sec.get("gate") or sec["kind"] == "compare"):
            note = sec["note"]
            assert "画像の判定器" in note or "シミュレータの正解" in note
            runs = [cfg["sources"][k]["run"] for k in sec.get("sources", [sec.get("source")])]
            if any("m4b_v2" not in r for r in runs):
                assert "シミュレータの正解" in note
            if any("m4b_v2" in r for r in runs):
                assert "画像の判定器 v2" in note
    assert "カメラ画像" in cfg["sections"][0]["title"]


def test_new_stamp_dir_never_reuses(tmp_path):
    now = datetime(2026, 10, 12, 1, 2, 3)
    a, b = vc.new_stamp_dir(tmp_path, now), vc.new_stamp_dir(tmp_path, now)
    assert a.name == "20261012_010203" and b.name == "20261012_010203_2"
