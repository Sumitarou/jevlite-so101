"""Teacher: a VLM on Ollama labels an image with one of the choices (offline use).

Calling convention (docs/spec.md): /api/chat, stream=false, think=false only for models that
support it, temperature 0, and a JSON Schema in `format` that limits `choice` to the enum.
Any failure returns choice=STOP with source="teacher_error" instead of raising (safe side).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import httpx
import numpy as np
from PIL import Image

from jevlite.api import Choice, Decision, JudgeStart, start_judging
from jevlite.config import load_config

ERROR_SOURCE = "teacher_error"


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": [c.name for c in Choice]},
            "reason": {"type": "string"},
        },
        "required": ["choice"],
    }


def parse_teacher_response(content: str | None) -> Choice | None:
    """Parse message.content. Returns None unless it is a JSON object with a valid choice."""
    if not content:
        return None
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    value = obj.get("choice")
    if not isinstance(value, str) or value not in Choice.__members__:
        return None
    return Choice[value]


def one_hot(choice: Choice) -> dict[str, float]:
    return {c.name: 1.0 if c is choice else 0.0 for c in Choice}


def encode_image(image: np.ndarray, fmt: str = "jpeg", quality: int = 90) -> str:
    buf = io.BytesIO()
    im = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    if fmt == "jpeg":
        im.save(buf, format="JPEG", quality=quality)
    else:
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OllamaTeacher:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.cfg = cfg or load_config("teacher.yaml")
        self.model = model or self.cfg["default_model"]
        self.url = self.cfg.get("ollama_url", "http://localhost:11434").rstrip("/")
        self.client = client or httpx.Client(timeout=float(self.cfg.get("timeout_s", 120)))
        self.source = f"teacher:{self.model}"
        self.last_error: str | None = None
        self.last_extra: dict[str, Any] = {}  # e.g. the model's reason, for logs/evaluation

    def build_request(self, image: np.ndarray) -> dict[str, Any]:
        img_b64 = encode_image(
            image, self.cfg.get("image_format", "jpeg"), int(self.cfg.get("jpeg_quality", 90))
        )
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": response_schema(),
            # temperature is fixed to 0 by convention; config cannot override it
            "options": {**self.cfg.get("options", {}), "temperature": 0},
            "messages": [{"role": "user", "content": self.cfg["prompt"], "images": [img_b64]}],
        }
        if self.model in self.cfg.get("think_models", []):
            body["think"] = False
        return body

    def _decision(self, choice: Choice, source: str, start: JudgeStart) -> Decision:
        return Decision(
            choice=choice,
            probs=one_hot(choice),
            t_mono=start.t_mono,
            latency_ms=start.latency_ms(),
            source=source,
        )

    def judge(self, image: np.ndarray, t_capture: float | None = None) -> Decision:
        """Decision.t_mono = t_capture (camera frame) or, if None, the time judging started.
        Errors also keep that t_mono (the STOP refers to the same image)."""
        start = start_judging(t_capture)
        self.last_error = None
        self.last_extra = {}
        try:
            resp = self.client.post(f"{self.url}/api/chat", json=self.build_request(image))
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
        except Exception as e:  # network, HTTP status, bad JSON body, missing keys
            self.last_error = f"{type(e).__name__}: {e}"
            return self._decision(Choice.STOP, ERROR_SOURCE, start)
        choice = self.decide(content)
        if choice is None:
            self.last_error = f"unparsable content: {content!r:.200}"
            return self._decision(Choice.STOP, ERROR_SOURCE, start)
        return self._decision(choice, self.source, start)

    def decide(self, content: str | None) -> Choice | None:
        """message.content -> Choice (None = unusable). Records the optional reason."""
        choice = parse_teacher_response(content)
        if choice is not None:
            reason = json.loads(content).get("reason")
            self.last_extra = {"reason": reason if isinstance(reason, str) else ""}
        return choice


# --- attribute mode: the model reports what it sees, the code picks the choice --------------


def attribute_schema() -> dict[str, Any]:
    """reason -> object_present -> touches_white, in this order (reason is generated first)."""
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "object_present": {"type": "boolean"},
            "touches_white": {"type": "boolean"},
        },
        "required": ["reason", "object_present", "touches_white"],
    }


def decide_from_attributes(obj: Any) -> tuple[Choice | None, bool]:
    """(choice, contradiction). object_present=false -> CONTINUE (A); true and not touching
    white -> SLOW (B); touches_white=true -> STOP (C), also when object_present=false (that
    contradiction is reported). Anything malformed -> (None, False)."""
    if not isinstance(obj, dict):
        return None, False
    present, touches = obj.get("object_present"), obj.get("touches_white")
    if not isinstance(present, bool) or not isinstance(touches, bool):
        return None, False
    if touches:
        return Choice.STOP, not present
    return (Choice.SLOW if present else Choice.CONTINUE), False


class OllamaAttrTeacher(OllamaTeacher):
    """Asks for attributes instead of a choice; optionally sends a reference image first
    ("the same table with nothing on it") and the image to judge second."""

    def __init__(self, cfg=None, model=None, client=None, reference: np.ndarray | None = None):
        super().__init__(cfg, model, client)
        self.reference = reference
        self._ref_b64 = None if reference is None else self._encode(reference)

    def _encode(self, image: np.ndarray) -> str:
        return encode_image(
            image, self.cfg.get("image_format", "jpeg"), int(self.cfg.get("jpeg_quality", 90))
        )

    def build_request(self, image: np.ndarray) -> dict[str, Any]:
        body = super().build_request(image)
        body["format"] = attribute_schema()
        if self._ref_b64 is not None:
            target = body["messages"][0]["images"][0]
            body["messages"][0]["images"] = [self._ref_b64, target]
        return body

    def decide(self, content: str | None) -> Choice | None:
        try:
            obj = json.loads(content or "")
        except (json.JSONDecodeError, TypeError):
            return None
        choice, contradiction = decide_from_attributes(obj)
        if choice is not None:
            self.last_extra = {
                "reason": obj.get("reason") if isinstance(obj.get("reason"), str) else "",
                "object_present": obj["object_present"],
                "touches_white": obj["touches_white"],
                "contradiction": contradiction,
            }
        return choice
