"""Round trip against the real Ollama server (excluded by default; run with -m ollama)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jevlite.api import Choice
from jevlite.teacher_ollama import OllamaTeacher

pytestmark = pytest.mark.ollama

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "synthetic" / "hand_near_a.png"


def test_roundtrip_default_model():
    img = np.asarray(Image.open(SAMPLE).convert("RGB"))
    teacher = OllamaTeacher()
    d = teacher.judge(img)
    assert teacher.last_error is None, teacher.last_error
    assert d.source == f"teacher:{teacher.model}"
    assert d.choice in set(Choice)
    assert sum(d.probs.values()) == 1.0
