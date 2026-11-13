"""Grab single frames from the Makkah live broadcast (Saudi Quran TV).

Primary source: SBA's own HLS stream of the official broadcast (SBA_HLS_URL / FALLBACK_HLS_URL, 1280x720; user
decision 2026-11-05, after YouTube's bot checks from 4 Nov 09:20 Makkah). Automatic backup: the YouTube live page
(1080p; same shots, graphics and crop as SBA, checked on simultaneous frames), used only when SBA gives no frame;
yt-dlp resolves the live page to a direct HLS URL, at most MAX_LOOKUPS_PER_DAY lookups a day. PRIMARY_SOURCE=youtube
in .env would put YouTube first again (it was primary from 2026-10-27 ~12:45 to 2026-11-05). FFmpeg pulls exactly one
frame per call and pipes it to Python as BMP bytes. Frames live in memory only. Nothing is written to disk unless
--save-samples is passed (test mode).

YouTube can be paused for some hours (--pause-youtube H; state "youtube_paused_until" in data/.ytdlp_state.json): no
lookups and no frames from YouTube until then, for every script (the live shot service's stream reader too).

YouTube is protected against "bots": every yt-dlp lookup hits its player API, and too many
lookups trigger "Sign in to confirm you're not a bot". So:
- the HLS URL is looked up once and reused across runs until it expires (~6 h) or FFmpeg
  can no longer read it (state in data/.ytdlp_state.json, incl. lookups per day);
- lookups are spaced at least 5 min apart and back off exponentially after failures
  (1 min doubling, 15 min doubling after a bot check, max 60 min);
- optional cookies (user decision) via YTDLP_COOKIES_BROWSER or YTDLP_COOKIES_FILE in .env;
- automatic fallback to SBA for any frame where YouTube fails.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yt_dlp
from dotenv import load_dotenv

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
STATE_PATH = PROJECT_ROOT / "data" / ".ytdlp_state.json"

DEFAULT_INTERVAL_S = 15.0
REFRESH_BEFORE_EXPIRY_S = 15 * 60
FALLBACK_TTL_S = 60 * 60  # if the URL has no expire= field
MIN_LOOKUP_SPACING_S = 5 * 60  # never two yt-dlp lookups closer than this
LOOKUP_BACKOFF_S = {"error": 60, "bot": 15 * 60}  # first wait after a failed lookup, doubles
MAX_LOOKUP_BACKOFF_S = 60 * 60
MAX_LOOKUPS_PER_DAY = 1  # user 2026-11-05: YouTube is the backup; one yt-dlp lookup a day at most (local date)
GRAB_TIMEOUT_S = 20  # a normal grab takes ~2 s; a stuck HLS read is killed and retried
MAX_BACKOFF_S = 300  # frame loop: exponential wait when every source failed
SBA_HLS_URL = "http://m.live.net.sa:1935/live/quran/playlist.m3u8"  # Saudi Quran TV from SBA's own server (720p)
# This stream offers video-only HLS formats (bv*), older ones combined (b).
# 1080p keeps small far-away people visible.
YTDL_FORMAT = "bv*[height<=1080][protocol^=m3u8]/b[height<=1080]/bv*/b"


class GrabError(RuntimeError):
    pass


class BotCheckError(GrabError):
    """YouTube asked to 'sign in to confirm you're not a bot' (too many lookups from this IP)."""


class LookupBackoff(GrabError):
    """A yt-dlp lookup is not allowed yet (spacing / backoff after failures)."""


@dataclass
class Frame:
    image: np.ndarray  # BGR, HxWx3 uint8
    captured_at: datetime  # UTC
    source: str = "youtube"  # "youtube" (1080p) or "sba" (SBA's HLS, 720p; "fallback" in rows before 2026-10-27)


def ffmpeg_exe() -> str:
    """System ffmpeg if on PATH, else the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Stream:
    url: str  # direct HLS manifest URL
    headers: dict[str, str]
    expires_at: float  # unix time

    def fresh(self) -> bool:
        return time.time() < self.expires_at - REFRESH_BEFORE_EXPIRY_S


def cookie_options() -> dict:
    """yt-dlp cookie options from .env (off unless the user sets one).

    YTDLP_COOKIES_BROWSER=firefox  (or firefox:<profile>) - read cookies from the browser
    YTDLP_COOKIES_FILE=<path>      - Netscape cookies.txt (keep it outside git)
    """
    if browser := os.getenv("YTDLP_COOKIES_BROWSER"):
        name, _, profile = browser.partition(":")
        return {"cookiesfrombrowser": (name.strip().lower(), profile.strip() or None, None, None)}
    if path := os.getenv("YTDLP_COOKIES_FILE"):
        return {"cookiefile": path}
    return {}


def resolve_stream(page_url: str) -> Stream:
    """Ask YouTube (via yt-dlp) for the direct HLS URL of a live page. Costs one player-API lookup.

    Use youtube_lookup() instead: it adds spacing, backoff and the daily counter.
    """
    opts = {
        "format": YTDL_FORMAT,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # YouTube needs a JS runtime; Node.js is installed on this PC (deno is the default).
        "js_runtimes": {"node": {}, "deno": {}},
        **cookie_options(),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(page_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        if "not a bot" in str(e) or "Sign in to confirm" in str(e):
            raise BotCheckError("YouTube bot check: 'Sign in to confirm you're not a bot'") from e
        raise GrabError(f"yt-dlp failed: {str(e)[:300]}") from e
    if not info.get("is_live", True):
        log.warning("Stream is not live right now (live_status=%s)", info.get("live_status"))
    url = info.get("url")
    if not url:
        raise GrabError("yt-dlp returned no direct URL")
    m = re.search(r"expire[/=](\d+)", url)
    expires_at = float(m.group(1)) if m else time.time() + FALLBACK_TTL_S
    log.info("Resolved stream: %sx%s format=%s, valid %.1f h", info.get("width"), info.get("height"),
             info.get("format_id"), (expires_at - time.time()) / 3600)
    return Stream(url, dict(info.get("http_headers") or {}), expires_at)


def load_state() -> dict:
    """yt-dlp state shared by all scripts: cached stream, lookups per day, backoff."""
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["calls"] = dict(sorted(state.get("calls", {}).items())[-30:])  # keep 30 days
    STATE_PATH.write_text(json.dumps(state, indent=1), encoding="utf-8")


def primary_source() -> str:
    """"sba" (default since 2026-11-05) or "youtube" (PRIMARY_SOURCE=youtube in .env)."""
    load_dotenv(PROJECT_ROOT / ".env")
    return "youtube" if os.getenv("PRIMARY_SOURCE", "sba").strip().lower() == "youtube" else "sba"


def sba_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("SBA_HLS_URL") or os.getenv("FALLBACK_HLS_URL") or SBA_HLS_URL


def youtube_paused(state: dict | None = None) -> float:
    """Seconds left of a YouTube pause (0: not paused)."""
    return max(0.0, (state or load_state()).get("youtube_paused_until", 0) - time.time())


def pause_youtube(hours: float) -> None:
    """No YouTube lookups or frames for `hours` (0: resume now), for every script."""
    state = load_state()
    state["youtube_paused_until"] = time.time() + hours * 3600 if hours > 0 else 0
    save_state(state)


def lookups_today(state: dict | None = None) -> int:
    return (state or load_state()).get("calls", {}).get(date.today().isoformat(), 0)


def cached_stream(page_url: str) -> Stream | None:
    """Stream URL saved by an earlier lookup, if it is for this page and not about to expire."""
    d = load_state().get("stream") or {}
    try:
        s = Stream(d["url"], d["headers"], d["expires_at"])
    except KeyError:
        return None
    return s if d.get("page_url") == page_url and s.fresh() else None


def youtube_lookup(page_url: str, reason: str) -> Stream:
    """One counted, rate-limited yt-dlp lookup. Raises LookupBackoff if not allowed yet."""
    state = load_state()
    if youtube_paused(state):
        raise LookupBackoff(f"YouTube paused for {youtube_paused(state) / 3600:.1f} more h")
    wait = state.get("next_lookup_after", 0) - time.time()
    if wait > 0:
        raise LookupBackoff(f"yt-dlp lookup blocked for {wait / 60:.1f} more min (backoff)")
    if lookups_today(state) >= MAX_LOOKUPS_PER_DAY:
        raise LookupBackoff(f"yt-dlp lookups today: {lookups_today(state)} of {MAX_LOOKUPS_PER_DAY} a day; next one tomorrow")
    today = date.today().isoformat()
    state.setdefault("calls", {})[today] = state.get("calls", {}).get(today, 0) + 1
    log.warning("yt-dlp lookup #%d today (%s)%s", state["calls"][today], reason,
                " with cookies" if cookie_options() else "")
    try:
        s = resolve_stream(page_url)
    except Exception as e:  # any failure counts, so a strange error can never cause rapid retries
        state["failures"] = state.get("failures", 0) + 1
        base = LOOKUP_BACKOFF_S["bot" if isinstance(e, BotCheckError) else "error"]
        delay = min(base * 2 ** (state["failures"] - 1), MAX_LOOKUP_BACKOFF_S)
        state["next_lookup_after"] = time.time() + delay
        save_state(state)
        log.error("%s. Next yt-dlp lookup in %.0f min.", e, delay / 60)
        raise
    state.update(failures=0, next_lookup_after=time.time() + MIN_LOOKUP_SPACING_S,
                 stream={"page_url": page_url, **asdict(s)})
    save_state(state)
    return s


def grab_frame(stream_url: str, headers: dict[str, str] | None = None, timeout: float = GRAB_TIMEOUT_S) -> np.ndarray:
    """Pull one decoded frame (BGR) from an HLS URL. Bytes go through a pipe, not disk."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += [
        "-live_start_index", "-1",  # newest segment = closest to "now"
        "-i", stream_url,
        "-frames:v", "1",
        "-an",
        "-f", "image2pipe", "-c:v", "bmp", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as e:
        raise GrabError(f"ffmpeg timed out after {timeout}s") from e
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode(errors="replace").strip()[-400:]
        raise GrabError(f"ffmpeg failed (code {proc.returncode}): {err}")
    image = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise GrabError("could not decode frame bytes")
    return image


class FrameGrabber:
    """Yields one frame every `interval_s` seconds: the primary source (SBA) first, the other one if it fails."""

    def __init__(self, page_url: str, interval_s: float = DEFAULT_INTERVAL_S, fallback_url: str | None = None,
                 primary: str = "sba"):
        self.page_url = page_url
        self.interval_s = interval_s
        self.fallback_url = fallback_url  # SBA's HLS URL
        self.primary = primary
        self._stale = False  # FFmpeg could not read the cached YouTube URL
        self._source: str | None = None

    @classmethod
    def from_env(cls, interval_s: float = DEFAULT_INTERVAL_S) -> "FrameGrabber":
        page = stream_url_from_env()
        return cls(page, interval_s, sba_url(), primary_source())

    def _youtube_stream(self) -> Stream:
        s = None if self._stale else cached_stream(self.page_url)
        if s is None:
            s = youtube_lookup(self.page_url, "URL stopped working" if self._stale else "no fresh cached URL")
            self._stale = False
        return s

    def _youtube_frame(self) -> np.ndarray:
        if youtube_paused():
            raise LookupBackoff(f"YouTube paused for {youtube_paused() / 3600:.1f} more h")
        s = self._youtube_stream()
        try:
            return grab_frame(s.url, s.headers)
        except GrabError:
            try:
                return grab_frame(s.url, s.headers)  # one retry, same URL, no lookup
            except GrabError:
                self._stale = True
                raise

    def grab(self) -> Frame:
        """Grab one frame now, from the primary source or (if that fails) the other one."""
        errors = []
        order = ["sba", "youtube"] if self.primary == "sba" else ["youtube", "sba"]
        image = source = None
        for src in order:
            if src == "sba" and not self.fallback_url:
                continue
            try:
                image = (grab_frame(self.fallback_url) if src == "sba"
                         else self._youtube_frame())
                source = src
                break
            except Exception as e:
                errors.append(f"{src}: {e}")
        if image is None:
            raise GrabError("; ".join(errors))
        if source != self._source:
            log.warning("Frame source: %s%s", source, f" ({errors[0]})" if errors else "")
            self._source = source
        return Frame(image=image, captured_at=datetime.now(timezone.utc), source=source)

    def frames(self, count: int | None = None) -> Iterator[Frame]:
        """Yield frames on a fixed schedule. Time spent by the consumer counts toward the interval.

        When every source fails, wait interval * 2^(n-1), max MAX_BACKOFF_S. YouTube lookups have
        their own spacing/backoff (youtube_lookup), so this loop never hammers YouTube.
        """
        yielded = 0
        failures = 0
        next_tick = time.monotonic()
        while count is None or yielded < count:
            try:
                frame = self.grab()
                failures = 0
            except Exception as e:  # network, yt-dlp, ffmpeg: keep running
                failures += 1
                wait = min(self.interval_s * 2 ** (failures - 1), MAX_BACKOFF_S)
                log.error("Frame grab failed (%d in a row), retry in %.0fs: %s", failures, wait, e)
                time.sleep(wait)
                next_tick = time.monotonic()
                continue
            yield frame
            yielded += 1
            next_tick += self.interval_s
            while next_tick <= time.monotonic():  # fell behind: skip missed ticks, keep the 15 s grid
                next_tick += self.interval_s
            time.sleep(next_tick - time.monotonic())


def save_sample(image: np.ndarray, captured_at: datetime, prefix: str = "frame") -> Path:
    """Test mode only: write one image to data/samples/."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"{prefix}_{captured_at:%Y%m%dT%H%M%SZ}.png"
    cv2.imwrite(str(path), image)
    return path


def stream_url_from_env() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("STREAM_URL")
    if not url:
        raise SystemExit("STREAM_URL is not set. Copy .env.example to .env.")
    return url


def check_cookies() -> None:
    """Read the configured cookies locally and report counts only (no values, no YouTube request)."""
    from yt_dlp.cookies import YoutubeDLCookieJar, extract_cookies_from_browser

    opts = cookie_options()
    if not opts:
        print("No cookies configured. Set YTDLP_COOKIES_BROWSER=firefox (or YTDLP_COOKIES_FILE) in .env.")
        return
    if "cookiesfrombrowser" in opts:
        name, profile, _, _ = opts["cookiesfrombrowser"]
        jar = extract_cookies_from_browser(name, profile)
        where = f"browser '{name}'" + (f", profile '{profile}'" if profile else "")
    else:
        jar = YoutubeDLCookieJar(opts["cookiefile"])
        jar.load(ignore_discard=True, ignore_expires=True)
        where = f"file {opts['cookiefile']}"
    yt = [c for c in jar if c.domain.endswith(("youtube.com", "google.com"))]
    logged_in = bool({c.name for c in yt} & {"SAPISID", "__Secure-3PSID", "SID"})
    print(f"Cookies from {where}: {len(jar)} total, {len(yt)} for youtube.com/google.com, "
          f"Google login cookie present: {logged_in}")
    if not logged_in:
        print("Not logged in: open youtube.com in that browser and sign in with the secondary account.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab frames from the Makkah live stream (memory only).")
    parser.add_argument("--count", type=int, default=3, help="number of frames (default 3)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="seconds between frames")
    parser.add_argument("--save-samples", action="store_true", help="TEST MODE: save frames to data/samples/")
    parser.add_argument("--check-cookies", action="store_true", help="check the cookie setup (no YouTube request)")
    parser.add_argument("--stats", action="store_true", help="show yt-dlp lookups per day and backoff state")
    parser.add_argument("--pause-youtube", type=float, metavar="H", help="no YouTube lookups or frames for H hours "
                        "(0: resume now)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_cookies:
        stream_url_from_env()  # loads .env
        check_cookies()
        return
    if args.pause_youtube is not None:
        pause_youtube(args.pause_youtube)
        print(f"YouTube paused for {youtube_paused() / 3600:.1f} h" if args.pause_youtube > 0 else "YouTube resumed")
        return
    if args.stats:
        st = load_state()
        print(f"yt-dlp lookups per day: {st.get('calls', {})}")
        print(f"consecutive failures: {st.get('failures', 0)}; next lookup allowed in "
              f"{max(0, st.get('next_lookup_after', 0) - time.time()) / 60:.1f} min")
        s = st.get("stream")
        print(f"cached URL valid for {(s['expires_at'] - time.time()) / 3600:.1f} h" if s else "no cached URL")
        print(f"primary source {primary_source()}; YouTube paused for {youtube_paused(st) / 3600:.1f} more h")
        return

    grabber = FrameGrabber.from_env(interval_s=args.interval)
    for i, frame in enumerate(grabber.frames(args.count), 1):
        h, w = frame.image.shape[:2]
        line = f"[{i}/{args.count}] {frame.captured_at:%H:%M:%S} UTC  {w}x{h}  source={frame.source}"
        if args.save_samples:
            line += f"  saved {save_sample(frame.image, frame.captured_at)}"
        print(line, flush=True)
        del frame  # memory only; dropped right after use
    print(f"yt-dlp lookups today: {lookups_today()}")


if __name__ == "__main__":
    main()
