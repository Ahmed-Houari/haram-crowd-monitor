"""One DM-Count on the GPU for the whole project (user request 2026-10-26: frame drops on this PC; no duplicate models).

The live shot service (scripts/live_shot.py) owns the model and also answers the crowd collector's predictions over a
local connection: 127.0.0.1 only, a random key per start in data/.model_share_key (HMAC challenge), raw bytes only (no
pickle). Frames stay in memory. The collector's SharedModel has CrowdModel's predict(); while the shot service is not
reachable it runs DM-Count on the CPU (slower; the same numbers up to fp16 rounding) and tries the shot service again
after RETRY_S.
"""
from __future__ import annotations

import logging
import secrets
import struct
import threading
import time
from multiprocessing.connection import Client, Listener

import numpy as np

from edge.runtime import PROJECT_ROOT

ADDRESS = ("127.0.0.1", 8765)
KEY_PATH = PROJECT_ROOT / "data" / ".model_share_key"
TIMEOUT_S = 30.0  # a request waits this long (it takes turns with the shot service's own frames)
RETRY_S = 60.0
log = logging.getLogger("model_share")


class LockedModel:
    """The model behind a lock: the shot service's frames and the collector's requests take turns on the GPU."""

    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    def predict(self, frame_bgr: np.ndarray):
        with self.lock:
            return self.model.predict(frame_bgr)

    def __getattr__(self, name):
        return getattr(self.model, name)


def serve(model: LockedModel) -> None:
    """Answer predictions in background threads (one per connection) for the life of the process."""
    key = secrets.token_bytes(32)
    tmp = KEY_PATH.with_suffix(".tmp")
    tmp.write_bytes(key)
    tmp.replace(KEY_PATH)
    listener = Listener(ADDRESS, authkey=key)

    def handle(conn) -> None:
        with conn:
            while True:
                try:
                    msg = conn.recv_bytes()
                    h, w = struct.unpack_from("<II", msg)
                    img = np.frombuffer(msg, np.uint8, offset=8).reshape(h, w, 3)
                    res = model.predict(img)
                    d = np.ascontiguousarray(res.density, np.float32)
                    conn.send_bytes(struct.pack("<IIdd", d.shape[0], d.shape[1], res.count, res.inference_ms) + d.tobytes())
                except (EOFError, OSError):
                    return
                except Exception:
                    log.exception("shared prediction failed")
                    return

    def accept() -> None:
        while True:
            try:
                conn = listener.accept()
            except Exception as e:  # a wrong key, a dropped connection
                log.warning("model share: connection refused (%s)", e)
                continue
            threading.Thread(target=handle, args=(conn,), daemon=True, name="model-share-conn").start()

    threading.Thread(target=accept, daemon=True, name="model-share").start()
    log.info("DM-Count shared with the crowd collector on %s:%d", *ADDRESS)


class SharedModel:
    """CrowdModel.predict() through the shot service's model; DM-Count on the CPU while that is not reachable."""

    def __init__(self):
        self.conn = None
        self.next_try = 0.0
        self.cpu = None

    def predict(self, frame_bgr: np.ndarray):
        from edge.crowd_model import CrowdResult

        if self.conn is None and time.monotonic() >= self.next_try:
            self._connect()
        if self.conn is not None and frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
            try:
                img = np.ascontiguousarray(frame_bgr, np.uint8)
                self.conn.send_bytes(struct.pack("<II", img.shape[0], img.shape[1]) + img.tobytes())
                if not self.conn.poll(TIMEOUT_S):
                    raise TimeoutError(f"no answer in {TIMEOUT_S:.0f} s")
                msg = self.conn.recv_bytes()
                dh, dw, count, ms = struct.unpack_from("<IIdd", msg)
                return CrowdResult(np.frombuffer(msg, np.float32, offset=24).reshape(dh, dw).copy(), count, ms)
            except Exception as e:
                log.warning("DM-Count via the shot service failed (%s); on the CPU, next try in %.0f s", e, RETRY_S)
                self._close()
        return self._cpu().predict(frame_bgr)

    def _connect(self) -> None:
        try:
            self.conn = Client(ADDRESS, authkey=KEY_PATH.read_bytes())
            log.info("DM-Count: the shot service's model (one copy on the GPU)")
        except Exception as e:
            self._close()
            log.warning("DM-Count: shot service not reachable (%s); on the CPU, next try in %.0f s", e, RETRY_S)

    def _close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None
        self.next_try = time.monotonic() + RETRY_S

    def _cpu(self):
        if self.cpu is None:
            from edge.crowd_model import CrowdModel

            self.cpu = CrowdModel(device="cpu", half=False)
        return self.cpu
