"""The broadcast as a continuous stream of frames for the live shot service (2026-10-26).

One FFmpeg process reads the live HLS stream and hands Python a decoded frame every 1 / FPS s of stream time
(rawvideo BGR on a pipe, frames in memory only, scaled to SIZE). The stream comes in 5 s segments, so frames arrive in
bursts: a reader thread queues them (stamped 1 / FPS s apart) and a consumer that falls behind skips to the newest ones.
Source: the primary one first (edge/frame_grabber.py primary_source; SBA's HLS stream since 2026-11-05, 720p scaled to
1080p), YouTube as the backup with the URL the crowd collector already looked up (data/.ytdlp_state.json): no extra
YouTube lookups, none while YouTube is paused. A source that gave no frame is skipped for SKIP_S, then tried again. When FFmpeg stops
(URL expired, network), the reader waits and starts again.
A stalled stream (2026-10-30 07:52 UTC: the SBA server kept the connection open and sent nothing; FFmpeg and the pipe
read waited for hours, the service looked alive and gave no readings): FFmpeg gets a read timeout (RW_TIMEOUT_S), and a
watchdog thread kills FFmpeg when no frame came for STALL_S, so the reader restarts either way.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from edge.frame_grabber import ffmpeg_exe, load_state, primary_source, sba_url, youtube_paused

log = logging.getLogger(__name__)

SIZE = (1920, 1080)
RETRY_S = (5, 120)
QUEUE = 16  # frames kept; a consumer more than KEEP_BEHIND behind jumps to the newest
KEEP_BEHIND = 8  # more than one 5 s segment: a burst is processed in full when the loop keeps up
SKIP_S = 120.0  # a source that gave no frame is not tried again for this long (the other one is used)
RW_TIMEOUT_S = 15  # FFmpeg gives up on a network read that gets nothing for this long
STALL_S = 120.0  # no frame for this long while FFmpeg runs: the watchdog kills it (the reader restarts)


@dataclass
class LiveFrame:
    image: np.ndarray
    captured_at: datetime  # arrival time (the stream runs ~10-20 s behind the broadcast, as for the collector)
    source: str = "youtube"  # or "sba"
    seq: int = 0


class StreamReader:
    def __init__(self, fps: float = 1.0):
        self.fps = fps
        self.queue: deque[LiveFrame] = deque(maxlen=QUEUE)
        self.cond = threading.Condition()
        self.seq = 0
        self.stop = False
        self.proc: subprocess.Popen | None = None
        self.source: str | None = None
        self.skip_until: dict[str, float] = {}
        self.last_frame = time.monotonic()  # last frame, or FFmpeg's start
        self.reading: str | None = None  # the source FFmpeg is reading now
        threading.Thread(target=self._run, name="stream-reader", daemon=True).start()
        threading.Thread(target=self._watchdog, name="stream-watchdog", daemon=True).start()

    def _watchdog(self) -> None:
        while not self.stop:
            time.sleep(10)
            p = self.proc
            if p is not None and p.poll() is None and time.monotonic() - self.last_frame > STALL_S:
                log.warning("stream (%s) stalled: no frame for %.0f s; killing FFmpeg", self.reading, time.monotonic() - self.last_frame)
                p.kill()

    def _url(self) -> tuple[str | None, str | None]:
        """(source, URL) to read now: the primary source unless it failed in the last SKIP_S, else the other."""
        order = ["sba", "youtube"] if primary_source() == "sba" else ["youtube", "sba"]
        for src in sorted(order, key=lambda s: time.time() < self.skip_until.get(s, 0)):
            if src == "sba":
                return src, sba_url()
            st = load_state()
            s = st.get("stream") or {}
            if not youtube_paused(st) and s.get("url") and s.get("expires_at", 0) > time.time() + 60:
                return src, s["url"]
        return None, None

    def _run(self) -> None:
        wait = RETRY_S[0]
        n = SIZE[0] * SIZE[1] * 3
        while not self.stop:
            src, url = self._url()
            if not url:
                log.info("no stream URL; retry in %d s", wait)
                time.sleep(wait)
                wait = min(wait * 2, RETRY_S[1])
                continue
            cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-rw_timeout", str(RW_TIMEOUT_S * 1_000_000), "-i", url, "-an",
                   "-vf", f"fps={self.fps},scale={SIZE[0]}:{SIZE[1]}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
            self.last_frame, self.reading = time.monotonic(), src
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=n,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            got, t0 = 0, None
            while not self.stop:
                buf = self.proc.stdout.read(n)
                if len(buf) < n:
                    break
                img = np.frombuffer(buf, np.uint8).reshape(SIZE[1], SIZE[0], 3)
                now = datetime.now(timezone.utc)
                # stream time: frames 1 / fps apart from the first one, never ahead of the clock (bursts per segment)
                t0 = t0 or now
                ts = min(t0 + timedelta(seconds=got / self.fps), now)
                with self.cond:
                    self.seq += 1
                    self.queue.append(LiveFrame(img, ts, src, seq=self.seq))
                    self.cond.notify_all()
                got += 1
                self.last_frame = time.monotonic()
                wait = RETRY_S[0]
                if src != self.source:
                    log.warning("stream source: %s", src)
                    self.source = src
            self.proc.kill()
            if not got:  # nothing from this source: the other one next time
                self.skip_until[src] = time.time() + SKIP_S
            log.warning("stream (%s) ended after %d frames; restarting in %d s", src, got, wait)
            time.sleep(wait)
            wait = min(wait * 2, RETRY_S[1])

    def next(self, after_seq: int, timeout: float = 10.0) -> LiveFrame | None:
        """The next frame after after_seq in stream order, or a newer one when the consumer is more than KEEP_BEHIND
        frames behind (waits up to timeout s)."""
        with self.cond:
            self.cond.wait_for(lambda: bool(self.queue) and self.queue[-1].seq > after_seq, timeout)
            newer = [f for f in self.queue if f.seq > after_seq]
            if not newer:
                return None
            return newer[-KEEP_BEHIND] if len(newer) > KEEP_BEHIND else newer[0]

    def close(self) -> None:
        self.stop = True
        if self.proc:
            self.proc.kill()
