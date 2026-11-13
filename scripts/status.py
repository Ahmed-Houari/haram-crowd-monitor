"""Simple status screen for the user ("Haram Monitor Status.bat" runs this). Read-only.

Shows RUNNING / STOPPED in big letters, last frame (Makkah time), frames accepted today per
camera, last TomTom poll, and whether any error was logged in the last hour. Never prints
anything from .env.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MAKKAH = timezone(timedelta(hours=3))
GREEN, RED, YELLOW, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"
FONT = {  # 5x5 block letters
    "A": [".###.", "#...#", "#####", "#...#", "#...#"], "D": ["####.", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "####.", "#....", "#####"], "G": [".####", "#....", "#..##", "#...#", ".####"],
    "I": ["#####", "..#..", "..#..", "..#..", "#####"], "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", ".###."], "P": ["####.", "#...#", "####.", "#....", "#...."],
    "R": ["####.", "#...#", "####.", "#..#.", "#...#"], "S": [".####", "#....", ".###.", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#.."], "U": ["#...#", "#...#", "#...#", "#...#", ".###."],
    " ": [".....", ".....", ".....", ".....", "....."],
}


def big(text: str, color: str) -> str:
    rows = ["  ".join(FONT[ch][r] for ch in text).replace("#", "█").replace(".", " ") for r in range(5)]
    return "\n".join(f"   {color}{row}{RESET}" for row in rows)


def makkah(ts: datetime | None) -> str:
    if ts is None:
        return "never"
    ago = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    when = f"{ago:.0f} min ago" if ago < 120 else f"{ago / 60:.1f} hours ago"
    return f"{ts.astimezone(MAKKAH):%Y-%m-%d %H:%M} Makkah time ({when})"


def errors_last_hour() -> list[str]:
    cutoff = datetime.now() - timedelta(hours=1)  # log files use local PC time
    found = []
    for name in ("supervisor", "crowd", "aux"):
        path = ROOT / "data" / "logs" / f"{name}.log"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-3000:]:
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m and (" ERROR " in line or " CRITICAL " in line) and datetime.fromisoformat(m.group(1)) >= cutoff:
                found.append(re.sub(r"key=[^&\s]+", "key=***", line))
    return found


def task_state() -> str:
    try:
        out = subprocess.run(["schtasks", "/query", "/tn", "HaramCrowdMonitor", "/fo", "csv", "/nh"],
                             capture_output=True, text=True, timeout=10, creationflags=0x08000000).stdout
        return out.strip().split(",")[-1].strip('"') if out.strip() else "not installed"
    except Exception:
        return "unknown"


def lan_ip() -> str:
    """This PC's address on the home network (the interface of the default route; no packet is sent)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET address: only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "<this PC's Wi-Fi IP>"


def main() -> None:
    os.system("")  # enable colors in the Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # block letters, also when piped
    from sqlalchemy import text

    from edge import db
    from edge.runtime import running_pid
    from edge.view_matcher import load_config

    now = datetime.now(timezone.utc)
    sup_alive = running_pid("supervisor") is not None
    try:
        st = json.loads((ROOT / "data" / ".crowd_status.json").read_text())
    except (OSError, ValueError):
        st = {}
    last_frame = datetime.fromisoformat(st["last_frame_utc"]) if st.get("last_frame_utc") else None
    fresh = last_frame is not None and now - last_frame < timedelta(minutes=3)

    print()
    if sup_alive:
        print(big("RUNNING", GREEN))
        if not fresh:
            print(f"\n   {YELLOW}Warning: the collector is running but no new frame for a while "
                  f"(check the internet connection).{RESET}")
    else:
        print(big("STOPPED", RED))
        print(f"\n   {RED}Collection is not running. It starts automatically when you log in to Windows,"
              f"\n   or within 5 minutes if the PC is on.{RESET}")
    print()
    print(f"   Autostart task:        {task_state()}")
    print("   Dashboard:             http://localhost:8000  (open it in your browser)")
    lan = "on" if running_pid("lan") else f"{YELLOW}off (starts with the collection){RESET}"
    print(f"   Dashboard on phone:    http://{lan_ip()}:8000  (same home Wi-Fi; LAN access {lan})")
    print(f"   Last video frame:      {makkah(last_frame)}")

    cfg = load_config()
    day_start = datetime.now(MAKKAH).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    with db.get_engine().connect() as c:
        q = lambda sql: c.execute(text(sql), {"d": day_start.strftime("%Y-%m-%d %H:%M:%S")}).all()  # noqa: E731
        acc = dict(q("SELECT view_id, COUNT(DISTINCT ts_utc) FROM crowd_frame WHERE accepted AND ts_utc >= :d GROUP BY view_id"))
        grabbed = q("SELECT COALESCE(SUM(frames), 0) FROM collector_minute WHERE minute_utc >= :d")[0][0]
        poll = q("SELECT MAX(poll_utc) FROM traffic_poll")[0][0]
        polls_today = q("SELECT COUNT(*) FROM traffic_poll WHERE poll_utc >= :d AND tiles_ok = tiles_total")[0][0]
    print(f"\n   Frames today (since midnight Makkah time): {grabbed:,} checked")
    for v in cfg["views"]:
        has_ref = (ROOT / v["ref_image"]).exists()
        note = {"low": "  (lower quality: count depends on zoom)", "untested": ""}.get(v.get("quality"), "")
        if not has_ref:
            note = "  (waiting for its reference picture)"
        print(f"     {v['id']:18s} accepted: {acc.get(v['id'], 0):>5,}{note}")
    if isinstance(poll, str):  # raw SQL on SQLite returns text
        poll = datetime.fromisoformat(poll)
    print(f"\n   Last traffic (TomTom) poll: {makkah(db.utc(poll) if poll else None)}   ({polls_today} good polls today)")
    errs = errors_last_hour()
    if errs:
        print(f"\n   Errors in the last hour: {RED}{BOLD}YES ({len(errs)}){RESET}")
        print(f"     latest: {errs[-1][:150]}")
        print("     details: data\\logs\\ (supervisor.log, crowd.log, aux.log)")
    else:
        print(f"\n   Errors in the last hour: {GREEN}{BOLD}NO{RESET}")
    print()


if __name__ == "__main__":
    main()
