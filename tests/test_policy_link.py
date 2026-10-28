"""scripts/sim/policy_link.py: the socket protocol between the evaluator and the policy server."""

import threading
from multiprocessing import AuthenticationError

import numpy as np
import pytest

from scripts.sim.policy_link import PolicyClient, PolicyError, decode, encode, new_key, serve


def test_encode_decode_round_trip_keeps_dtype_and_shape():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    msg = {"cmd": "act", "image": img, "state": np.array([0.1, -0.2], np.float32),
           "nested": [np.float64(1.5), {"q": np.zeros((2, 2))}]}  # fmt: skip
    out = decode(encode(msg))
    assert out["image"].dtype == np.uint8 and out["image"].shape == (2, 3, 3)
    np.testing.assert_array_equal(out["image"], img)
    assert out["state"].dtype == np.float32
    assert out["nested"][0] == 1.5 and out["nested"][1]["q"].shape == (2, 2)
    # nothing numpy-specific is left for pickle to carry between numpy versions
    enc = encode(msg)
    assert isinstance(enc["image"]["data"], bytes) and isinstance(enc["nested"][0], float)


@pytest.fixture
def server(tmp_path):
    addr, key = str(tmp_path / "p.sock"), new_key()
    calls = []

    def handle(msg):
        calls.append(msg["cmd"])
        if msg["cmd"] == "reset":
            return {"queue_len": 0}
        if msg["cmd"] == "act":
            return {"action": msg["state"] * 2, "names": ["a", "b"]}
        raise ValueError("unknown")

    ready = threading.Event()
    th = threading.Thread(target=serve, args=(addr, key, handle, ready.set), daemon=True)
    th.start()
    assert ready.wait(5)
    yield addr, key, calls, th


def test_client_server_round_trip_and_close(server):
    addr, key, calls, th = server
    c = PolicyClient.connect(addr, key, timeout_s=5)
    assert c.reset()["queue_len"] == 0
    r = c.act(np.zeros((4, 4, 3), np.uint8), np.array([1.0, 2.0]))
    np.testing.assert_allclose(r["action"], [2.0, 4.0])
    with pytest.raises(PolicyError, match="unknown"):
        c.request({"cmd": "nope"})
    c.close()
    th.join(5)
    assert not th.is_alive()
    assert calls == ["reset", "act", "nope"]


def test_wrong_key_is_refused_and_the_server_keeps_waiting(server):
    addr, key, calls, th = server
    with pytest.raises(AuthenticationError):
        PolicyClient.connect(addr, new_key(), timeout_s=2)
    c = PolicyClient.connect(addr, key, timeout_s=5)
    assert c.reset()["queue_len"] == 0 and calls == ["reset"]
    c.close()
    th.join(5)
    assert not th.is_alive()


def test_connect_gives_up_when_the_server_died(tmp_path):
    with pytest.raises(FileNotFoundError):
        PolicyClient.connect(str(tmp_path / "none.sock"), new_key(), timeout_s=30,
                             alive=lambda: False)  # fmt: skip
