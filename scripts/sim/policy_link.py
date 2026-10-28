"""Link between the evaluator (system Python, rclpy) and the policy server (train/ env, torch).

The two run in different Python environments (ROS' numpy 1.26 vs lerobot's numpy 2.2), so they
talk over a local socket (multiprocessing.connection, AF_UNIX, with an authkey). Messages are
plain dicts; numpy arrays travel as {dtype, shape, bytes} so neither side unpickles the other's
numpy objects. Requests and replies:

  {"cmd": "reset"}                         -> {"ok": True, "queue_len": 0}
  {"cmd": "act", "image": HxWx3 uint8, "state": (J,) float}
                                           -> {"ok": True, "action": (J,), "command": (6,),
                                               "names": [...], "infer_ms": float,
                                               "inferred": bool, "queue_len": int}
  {"cmd": "info"}                          -> {"ok": True, ...server settings...}
  {"cmd": "close"}                         -> {"ok": True} and the server stops
A failed request gets {"ok": False, "error": "..."}; the client raises PolicyError.
"""

from __future__ import annotations

import os
import secrets
import tempfile
import time
from collections.abc import Callable
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path

import numpy as np

KEY_ENV = "JEVLITE_POLICY_KEY"  # hex authkey, passed to the server through its environment


class PolicyError(RuntimeError):
    pass


def default_address(name: str = "jevlite_policy") -> str:
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return str(Path(base) / f"{name}_{os.getpid()}.sock")


def new_key() -> str:
    return secrets.token_hex(16)


def encode(obj):
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {"__nd__": True, "dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [encode(v) for v in obj]
    return obj


def decode(obj):
    if isinstance(obj, dict):
        if obj.get("__nd__"):
            return np.frombuffer(obj["data"], np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


class PolicyClient:
    def __init__(self, conn: Connection):
        self.conn = conn

    @classmethod
    def connect(cls, address: str, key: str, timeout_s: float = 60.0, alive=None) -> PolicyClient:
        """Retry until the server listens (it may still be loading the model).

        alive() -> bool lets the caller stop early when the server process has died.
        """
        end = time.monotonic() + timeout_s
        while True:
            try:
                return cls(Client(address, family="AF_UNIX", authkey=bytes.fromhex(key)))
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > end or (alive is not None and not alive()):
                    raise
                time.sleep(0.2)

    def request(self, msg: dict) -> dict:
        self.conn.send(encode(msg))
        rep = decode(self.conn.recv())
        if not rep.get("ok"):
            raise PolicyError(rep.get("error", "unknown error"))
        return rep

    def reset(self) -> dict:
        return self.request({"cmd": "reset"})

    def act(self, image: np.ndarray, state: np.ndarray) -> dict:
        return self.request({"cmd": "act", "image": image, "state": np.asarray(state, np.float32)})

    def info(self) -> dict:
        return self.request({"cmd": "info"})

    def close(self) -> None:
        try:
            self.request({"cmd": "close"})
        except (EOFError, OSError, PolicyError):
            pass
        self.conn.close()


def serve(address: str, key: str, handle: Callable[[dict], dict], on_ready=None) -> None:
    """Accept one client and answer its requests with handle(msg) until "close" or EOF.
    A client with the wrong key is turned away and the server keeps waiting."""
    if Path(address).exists():
        Path(address).unlink()  # a stale socket file from a killed server
    with Listener(address, family="AF_UNIX", authkey=bytes.fromhex(key)) as lis:
        if on_ready:
            on_ready()
        while True:
            try:
                conn = lis.accept()
                break
            except (AuthenticationError, EOFError, ConnectionError):
                continue
        with conn:
            while True:
                try:
                    msg = decode(conn.recv())
                except EOFError:
                    return
                if msg.get("cmd") == "close":
                    conn.send({"ok": True})
                    return
                try:
                    rep = {"ok": True, **handle(msg)}
                except Exception as e:  # reported to the client, the server keeps going
                    rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                conn.send(encode(rep))
