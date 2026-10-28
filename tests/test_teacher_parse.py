import base64
import io
import json
import time

import httpx
import numpy as np
import pytest
from PIL import Image

from jevlite.api import Choice
from jevlite.config import load_config
from jevlite.teacher_ollama import (
    ERROR_SOURCE,
    OllamaTeacher,
    parse_teacher_response,
    response_schema,
)

IMG = np.zeros((32, 48, 3), dtype=np.uint8)
CFG = {
    "ollama_url": "http://ollama.test",
    "default_model": "gemma3:4b",
    "think_models": ["qwen3.5:27b"],
    "options": {"temperature": 0.7, "num_ctx": 4096},
    "prompt": "judge",
}


# --- parse_teacher_response -------------------------------------------------------------


@pytest.mark.parametrize("name", ["CONTINUE", "SLOW", "STOP"])
def test_parse_valid(name):
    assert parse_teacher_response(json.dumps({"choice": name, "reason": "x"})) is Choice[name]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        "not json",
        '{"choice": "STOP"',  # truncated
        '["STOP"]',
        '"STOP"',
        "{}",
        '{"choice": null}',
        '{"choice": 2}',
        '{"choice": "FAST"}',
        '{"choice": "stop"}',  # enum is exact; schema forbids other spellings
        '{"answer": "STOP"}',
    ],
)
def test_parse_invalid_returns_none(content):
    assert parse_teacher_response(content) is None


def test_schema_limits_choice_to_enum():
    s = response_schema()
    assert s["properties"]["choice"]["enum"] == ["CONTINUE", "SLOW", "STOP"]
    assert s["required"] == ["choice"]


# --- request building --------------------------------------------------------------------


def test_request_follows_calling_convention():
    body = OllamaTeacher(CFG).build_request(IMG)
    assert body["model"] == "gemma3:4b"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0  # config cannot override
    assert body["options"]["num_ctx"] == 4096
    assert body["format"] == response_schema()
    assert "think" not in body  # gemma3 does not support thinking
    msg = body["messages"][0]
    assert msg["content"] == "judge"
    decoded = Image.open(io.BytesIO(base64.b64decode(msg["images"][0])))
    assert decoded.size == (48, 32)


def test_think_false_only_for_supported_models():
    assert OllamaTeacher(CFG, model="qwen3.5:27b").build_request(IMG)["think"] is False
    assert "think" not in OllamaTeacher(CFG, model="qwen2.5vl:7b").build_request(IMG)


def test_repo_teacher_config_is_valid():
    cfg = load_config("teacher.yaml")
    assert cfg["default_model"] in cfg["models"]
    assert set(cfg["think_models"]) <= set(cfg["models"])
    assert cfg["prompt"].strip()


# --- judge() with a mocked transport ---------------------------------------------------


def make_teacher(handler, model=None):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaTeacher(CFG, model=model, client=client)


def chat_response(content):
    return lambda req: httpx.Response(200, json={"message": {"content": content}})


def test_judge_success():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": '{"choice":"SLOW"}'}}
        )

    t = make_teacher(handler)
    d = t.judge(IMG)
    assert seen["url"] == "http://ollama.test/api/chat"
    assert seen["body"]["stream"] is False
    assert d.choice is Choice.SLOW
    assert d.probs == {"CONTINUE": 0.0, "SLOW": 1.0, "STOP": 0.0}
    assert d.source == "teacher:gemma3:4b"
    assert d.latency_ms >= 0
    assert t.last_error is None


def _raise(exc):
    def handler(req):
        raise exc

    return handler


@pytest.mark.parametrize(
    "handler",
    [
        chat_response('{"choice":"FAST"}'),  # out of enum
        chat_response("garbage"),  # broken JSON
        chat_response(""),  # empty content
        lambda req: httpx.Response(200, json={"error": "no message"}),  # missing keys
        lambda req: httpx.Response(200, content=b"<html>"),  # body is not JSON
        lambda req: httpx.Response(500, json={"error": "boom"}),  # HTTP error
        lambda req: httpx.Response(404, json={"error": "model not found"}),
        _raise(httpx.ConnectError("refused")),  # Ollama down
        _raise(httpx.ReadTimeout("timeout")),  # timeout
    ],
    ids=[
        "enum",
        "broken-json",
        "empty",
        "missing-keys",
        "non-json-body",
        "http500",
        "http404",
        "connect",
        "timeout",
    ],
)
def test_judge_failure_is_stop_not_exception(handler):
    t = make_teacher(handler)
    d = t.judge(IMG)
    assert d.choice is Choice.STOP
    assert d.source == ERROR_SOURCE
    assert d.probs == {"CONTINUE": 0.0, "SLOW": 0.0, "STOP": 1.0}
    assert t.last_error


# --- t_mono / latency_ms -----------------------------------------------------------------


def slow_ok(req):
    time.sleep(0.05)
    return httpx.Response(200, json={"message": {"content": '{"choice":"SLOW"}'}})


def slow_fail(req):
    time.sleep(0.05)
    raise httpx.ReadTimeout("timeout")


@pytest.mark.parametrize("handler", [slow_ok, slow_fail], ids=["ok", "error"])
def test_teacher_t_mono_is_the_given_capture_time(handler):
    t = make_teacher(handler)
    t_capture = time.monotonic() - 0.5
    d = t.judge(IMG, t_capture=t_capture)
    assert d.t_mono == t_capture  # also on errors: the STOP refers to that image
    assert 50.0 <= d.latency_ms < 500.0  # request time only, not the 0.5 s image age


@pytest.mark.parametrize("handler", [slow_ok, slow_fail], ids=["ok", "error"])
def test_teacher_t_mono_without_capture_time_is_the_start_time(handler):
    t = make_teacher(handler)
    before = time.monotonic()
    d = t.judge(IMG)
    after = time.monotonic()
    assert before <= d.t_mono <= before + 0.01
    assert after - d.t_mono >= 0.05


# --- attribute mode (OllamaAttrTeacher) ---------------------------------------------------

from jevlite.teacher_ollama import (  # noqa: E402
    OllamaAttrTeacher,
    attribute_schema,
    decide_from_attributes,
)


@pytest.mark.parametrize(
    ("present", "touches", "choice", "contradiction"),
    [
        (False, False, Choice.CONTINUE, False),  # A
        (True, False, Choice.SLOW, False),  # B
        (True, True, Choice.STOP, False),  # C
        (False, True, Choice.STOP, True),  # contradiction -> the safe side, and reported
    ],
)
def test_choice_is_decided_from_attributes(present, touches, choice, contradiction):
    obj = {"reason": "x", "object_present": present, "touches_white": touches}
    assert decide_from_attributes(obj) == (choice, contradiction)


@pytest.mark.parametrize(
    "obj",
    [None, [], {}, {"object_present": True}, {"object_present": "yes", "touches_white": False},
     {"object_present": True, "touches_white": 1}],
)  # fmt: skip
def test_malformed_attributes_are_rejected(obj):
    assert decide_from_attributes(obj) == (None, False)


def test_attribute_schema_order_and_required():
    s = attribute_schema()
    assert list(s["properties"]) == ["reason", "object_present", "touches_white"]
    assert s["required"] == ["reason", "object_present", "touches_white"]
    assert s["properties"]["object_present"]["type"] == "boolean"


def attr_teacher(content, reference=None, seen=None):
    def handler(req):
        if seen is not None:
            seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"message": {"content": content}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaAttrTeacher(CFG, client=client, reference=reference)


def test_attr_teacher_decides_in_code_and_records_attributes():
    seen = {}
    t = attr_teacher(
        '{"reason": "a hand on the green band", "object_present": true, "touches_white": false}',
        seen=seen,
    )
    d = t.judge(IMG)
    assert d.choice is Choice.SLOW and d.source == "teacher:gemma3:4b"
    assert t.last_extra == {
        "reason": "a hand on the green band", "object_present": True,
        "touches_white": False, "contradiction": False,
    }  # fmt: skip
    assert seen["body"]["format"] == attribute_schema()
    assert len(seen["body"]["messages"][0]["images"]) == 1


def test_attr_teacher_contradiction_is_stop_and_recorded():
    t = attr_teacher('{"reason": "?", "object_present": false, "touches_white": true}')
    assert t.judge(IMG).choice is Choice.STOP
    assert t.last_extra["contradiction"] is True


@pytest.mark.parametrize(
    "content", ['{"choice": "CONTINUE"}', "not json", '{"object_present": true}']
)
def test_attr_teacher_unusable_answer_is_teacher_error_stop(content):
    t = attr_teacher(content)
    d = t.judge(IMG)
    assert d.choice is Choice.STOP and d.source == ERROR_SOURCE
    assert t.last_error


def test_reference_image_is_sent_first_then_the_target():
    ref = np.full((32, 48, 3), 200, np.uint8)
    seen = {}
    t = attr_teacher(
        '{"reason": "", "object_present": false, "touches_white": false}', reference=ref, seen=seen
    )
    t.judge(IMG)
    images = seen["body"]["messages"][0]["images"]
    assert len(images) == 2
    first = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[0]))))
    second = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[1]))))
    assert first.mean() > 150 and second.mean() < 50  # reference (bright) first, IMG (black) second


def test_choice_teacher_records_its_optional_reason():
    t = make_teacher(chat_response('{"choice": "STOP", "reason": "hand on white"}'))
    t.judge(IMG)
    assert t.last_extra == {"reason": "hand on white"}
