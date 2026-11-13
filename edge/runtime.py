"""Small helpers for long-running processes: single-instance lock, file logging, keep-awake."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOCK_DIR = PROJECT_ROOT / "data"
ALREADY_RUNNING_EXIT = 3  # exit code when another instance holds the lock

_locks: dict[str, object] = {}


def single_instance(name: str) -> bool:
    """Take an exclusive OS lock on data/.lock_<name>. False if another process holds it.

    The lock is released automatically when the process exits (even after a crash).
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_DIR / f".lock_{name}", "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return False
    _locks[name] = f  # keep the handle open for the life of the process
    # the locked byte cannot be read by other processes, so the PID goes to a separate file
    (LOCK_DIR / f".pid_{name}").write_text(str(os.getpid()))
    return True


def running_pid(name: str) -> int | None:
    """PID of the process holding the `name` lock, if it is still alive (for status screens)."""
    try:
        pid = int((LOCK_DIR / f".pid_{name}").read_text().strip())
    except (OSError, ValueError):
        return None
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            return None
    import ctypes

    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return pid if code.value == 259 else None  # STILL_ACTIVE


def setup_logging(name: str, level: int = logging.INFO) -> None:
    """Log to the console and to data/logs/<name>.log (5 MB x 5 files)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    handlers = [logging.handlers.RotatingFileHandler(LOG_DIR / f"{name}.log", maxBytes=5_000_000, backupCount=5,
                                                     encoding="utf-8")]
    if sys.stderr is not None:  # pythonw.exe (Task Scheduler) has no console
        handlers.append(logging.StreamHandler())
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("urllib3").setLevel(logging.INFO)  # never DEBUG: it logs full URLs (TomTom key)


def on_ac_power() -> bool | None:
    """True on AC, False on battery, None if unknown (Windows only)."""
    if sys.platform != "win32":
        return None
    import ctypes

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    s = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return None
    return {0: False, 1: True}.get(s.ACLineStatus)


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep on idle (display may still turn off). Lid close and the power
    button still put the PC to sleep: that is a Windows power setting the user controls."""
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def private_memory_mb(pid: int | None = None) -> float | None:
    """Private (committed) memory of a process in MB, None if unknown (Windows only; no psutil in the venv)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]\
            + [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                              "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                              "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                              "PrivateUsage")]

    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.OpenProcess.restype = k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.GetCurrentProcess() if pid is None else k32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED | VM_READ
    if not h:
        return None
    try:
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return None
        return c.PrivateUsage / 2 ** 20
    finally:
        if pid is not None:
            k32.CloseHandle(h)


def child_pids(parent: int) -> list[int]:
    """PIDs whose parent is `parent` (the venv's python.exe launcher starts the real interpreter as a child)."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class PE32(ctypes.Structure):  # PROCESSENTRY32W
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    out = []
    try:
        e = PE32()
        e.dwSize = ctypes.sizeof(PE32)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            if e.th32ParentProcessID == parent:
                out.append(int(e.th32ProcessID))
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def terminate_pid(pid: int) -> bool:
    """Kill a process by PID (Windows); True if the kill was issued."""
    if sys.platform != "win32":
        try:
            os.kill(pid, 15)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if not h:
        return False
    try:
        return bool(k32.TerminateProcess(h, 1))
    finally:
        k32.CloseHandle(h)
