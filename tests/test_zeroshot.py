import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jevlite.api import Choice
from jevlite.config import load_config
from jevlite.zeroshot import ZeroShotJudge, probs_from_embeddings, softmax
from scripts import run_zeroshot_on_images as runner

CHOICES = {
    "choices": [
        {"name": "CONTINUE", "prompts": ["c"]},
        {"name": "SLOW", "prompts": ["s1", "s2"]},
        {"name": "STOP", "prompts": ["x"]},
    ]
}


class FakeBackend:
    """Image embedding = unit vector chosen by the mean pixel value (0 / 1 / 2)."""

    logit_scale = 10.0

    def __init__(self):
        self.text_calls = []
        self.image_calls = 0

    def text_embeddings(self, prompts):
        self.text_calls.append(prompts)
        return np.eye(3)

    def image_embedding(self, image):
        self.image_calls += 1
        v = np.zeros(3)
        v[int(image.mean()) % 3] = 1.0
        return v


def test_softmax_sums_to_one_and_is_stable():
    p = softmax(np.array([1000.0, 1000.0, 999.0]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[1])


def test_probs_from_embeddings():
    probs = probs_from_embeddings(np.array([0, 1.0, 0]), np.eye(3), 10.0, ["A", "B", "C"])
    assert max(probs, key=probs.get) == "B"
    assert sum(probs.values()) == pytest.approx(1.0)


def test_judge_returns_decision_and_runs_warmup():
    be = FakeBackend()
    judge = ZeroShotJudge(be, CHOICES, warmup=2)
    assert be.text_calls == [[["c"], ["s1", "s2"], ["x"]]]
    assert be.image_calls == 2
    d = judge.judge(np.full((8, 8, 3), 2, dtype=np.uint8))
    assert d.choice is Choice.STOP
    assert d.source == "v0"
    assert d.latency_ms >= 0
    assert set(d.probs) == {"CONTINUE", "SLOW", "STOP"}


def test_softmax_scale_overrides_model_scale():
    assert ZeroShotJudge(FakeBackend(), CHOICES).scale == 10.0
    assert ZeroShotJudge(FakeBackend(), CHOICES, softmax_scale=3.0).scale == 3.0


def test_choices_must_cover_all():
    bad = {"choices": [{"name": "CONTINUE", "prompts": ["c"]}, {"name": "STOP", "prompts": ["x"]}]}
    with pytest.raises(ValueError):
        ZeroShotJudge(FakeBackend(), bad)


def test_repo_configs_are_valid():
    ZeroShotJudge(FakeBackend(), load_config("choices.yaml"))
    z = load_config("zeroshot.yaml")
    assert z["model_id"] and len(z["revision"]) == 40


def test_latency_stats():
    s = runner.latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["n"] == 5 and s["p50"] == 3.0 and s["max"] == 100.0
    assert 4.0 < s["p95"] <= 100.0
    assert runner.latency_stats([]) == {"n": 0}


def test_runner_on_folder(tmp_path: Path):
    for i, sub in enumerate(["a", "b/c"]):
        (tmp_path / sub).mkdir(parents=True)
        Image.new("RGB", (16, 16), (i, i, i)).save(tmp_path / sub / f"img{i}.png")
    (tmp_path / "notes.txt").write_text("skip me")
    out = io.StringIO()
    csv_path = tmp_path / "out" / "v0.csv"
    stats = runner.run(ZeroShotJudge(FakeBackend(), CHOICES), tmp_path, csv_path, out=out)
    text = out.getvalue()
    assert stats["n"] == 2
    assert "a/img0.png" in text and "CONTINUE" in text
    assert "b/c/img1.png" in text and "SLOW" in text
    assert "p95=" in text
    assert csv_path.read_text().splitlines()[0] == "path,choice,CONTINUE,SLOW,STOP,latency_ms"


def test_synthetic_samples_are_deterministic():
    from scripts.make_synthetic_samples import SCENES, make

    for i, name in enumerate(SCENES):
        a, b = np.asarray(make(name, i)), np.asarray(make(name, i))
        assert a.shape == (240, 320, 3)
        assert np.array_equal(a, b)


@pytest.mark.gpu
def test_siglip_real_model_on_synthetic_sample():
    from jevlite.zeroshot import build_zeroshot_judge

    img = np.asarray(Image.open(Path("samples/synthetic/hand_near_a.png")).convert("RGB"))
    d = build_zeroshot_judge().judge(img)
    assert sum(d.probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert d.source == "v0"
    assert d.latency_ms < 1000


# --- t_mono / latency_ms -----------------------------------------------------------------


class SlowBackend(FakeBackend):
    def image_embedding(self, image):
        time.sleep(0.05)
        return super().image_embedding(image)


IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def test_v0_t_mono_is_the_given_capture_time():
    judge = ZeroShotJudge(SlowBackend(), CHOICES)
    t_capture = time.monotonic() - 0.123
    d = judge.judge(IMG, t_capture=t_capture)
    assert d.t_mono == t_capture
    assert 50.0 <= d.latency_ms < 500.0  # judging time only, not the 123 ms image age


def test_v0_t_mono_without_capture_time_is_the_start_time():
    judge = ZeroShotJudge(SlowBackend(), CHOICES)
    before = time.monotonic()
    d = judge.judge(IMG)
    after = time.monotonic()
    assert before <= d.t_mono <= before + 0.01  # start of judging...
    assert after - d.t_mono >= 0.05  # ...not the finish (50 ms later)
    assert d.latency_ms >= 50.0


def test_v0_judge_frame_helper():
    from jevlite.api import judge_frame

    class F:
        rgb = IMG
        t_capture = 7.5

    assert judge_frame(ZeroShotJudge(FakeBackend(), CHOICES), F()).t_mono == 7.5
