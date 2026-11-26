#!/usr/bin/env python3
"""
Automated benchmark harness for VIctory (Xbox Series compatibility layer).

Steps performed deterministically on every run:
1. Build: `cargo build --release` (abort on compilation failure).
2. Launch: Start the title via `target/release/victory.exe` with per-frame
   `Present()` telemetry logging (`VICTORY_FRAMETIMES=logs/frametimes.csv`).
3. Auto-Input: Wait for the game window to appear and gain focus, then send
   the `Enter` key using Win32 `SendInput` via `ctypes`.
4. Scene Load Wait: Wait 120 seconds (2 minutes) for the game to finish
   loading into the 3D scene.
5. Measure FPS: Sample real `Present()` frame times over a 20-second
   measurement window (seconds 120-140) and report `avg_fps`, `p1_low_fps`,
   and `frame_count`.
6. Clean Teardown: Force-terminate the game process and any child processes.

A hard 180-second timeout kills deadlocked/hung runs and exits non-zero.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
FRAMETIMES_CSV = LOGS_DIR / "frametimes.csv"
RUN_LOG = LOGS_DIR / "run.log"
DEFAULT_GAME_DIR = Path(r"K:\GAME OFFLINE\Rockstar\GTA VI\GTA6.Xbox.v1.0.2.1-CYBERLEEK")
HARD_TIMEOUT_SEC = 180.0
SCENE_LOAD_WAIT_SEC = 120.0
MEASURE_WINDOW_SEC = 20.0


# ---------------------------------------------------------------------------
# Win32 ctypes structures & helpers (64-bit safe)
# ---------------------------------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

VK_RETURN = 0x0D
MAPVK_VK_TO_VSC = 0
SW_RESTORE = 9
SW_SHOW = 5

ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUT_UNION),
    ]


assert ctypes.sizeof(INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28), (
    f"Unexpected INPUT struct size: {ctypes.sizeof(INPUT)}"
)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
user32.AllowSetForegroundWindow.restype = wintypes.BOOL
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
kernel32.GetCurrentThreadId.argtypes = []
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


def find_game_window(pid: int) -> Optional[Tuple[int, str]]:
    """Return (hwnd, title) of a visible top-level window belonging to `pid`."""
    found: List[Tuple[int, str, int]] = []

    @WNDENUMPROC
    def _cb(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        win_pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(win_pid))
        if win_pid.value != pid:
            return True
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w < 200 or h < 200:
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        found.append((hwnd, buf.value, w * h))
        return True

    user32.EnumWindows(_cb, 0)
    if not found:
        return None
    # Prefer the largest visible window (the 3D swap-chain window).
    found.sort(key=lambda item: item[2], reverse=True)
    return found[0][0], found[0][1]


def focus_window(hwnd: int, pid: int) -> bool:
    """Bring `hwnd` to the foreground so DirectInput/RawInput/GameInput sees keys."""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    else:
        user32.ShowWindow(hwnd, SW_SHOW)

    user32.AllowSetForegroundWindow(pid)

    def _is_focused() -> bool:
        fg = user32.GetForegroundWindow()
        if not fg:
            return False
        if fg == hwnd:
            return True
        fg_pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(fg, ctypes.byref(fg_pid))
        return fg_pid.value == pid

    if _is_focused():
        return True

    # Tap Alt (KEYEVENTF_EXTENDEDKEY) to release Windows foreground lock if needed.
    alt_down = INPUT()
    alt_down.type = INPUT_KEYBOARD
    alt_down.ki = KEYBDINPUT(wVk=0x12, wScan=0x38, dwFlags=0, time=0, dwExtraInfo=0)
    alt_up = INPUT()
    alt_up.type = INPUT_KEYBOARD
    alt_up.ki = KEYBDINPUT(wVk=0x12, wScan=0x38, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
    user32.SendInput(1, ctypes.byref(alt_down), ctypes.sizeof(INPUT))
    user32.SendInput(1, ctypes.byref(alt_up), ctypes.sizeof(INPUT))

    fg_hwnd = user32.GetForegroundWindow()
    cur_tid = kernel32.GetCurrentThreadId()
    fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
    tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)

    attached_fg = False
    attached_tgt = False
    try:
        if fg_tid and fg_tid != cur_tid:
            attached_fg = bool(user32.AttachThreadInput(cur_tid, fg_tid, True))
        if tgt_tid and tgt_tid != cur_tid:
            attached_tgt = bool(user32.AttachThreadInput(cur_tid, tgt_tid, True))
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached_tgt:
            user32.AttachThreadInput(cur_tid, tgt_tid, False)
        if attached_fg:
            user32.AttachThreadInput(cur_tid, fg_tid, False)

    time.sleep(0.05)
    return _is_focused()


def send_enter_key(hold_sec: float = 0.25) -> bool:
    """Send Enter key press & release via Win32 SendInput (with hardware scancode)."""
    scancode = user32.MapVirtualKeyW(VK_RETURN, MAPVK_VK_TO_VSC) or 0x1C

    down = INPUT()
    down.type = INPUT_KEYBOARD
    down.ki = KEYBDINPUT(
        wVk=VK_RETURN,
        wScan=scancode,
        dwFlags=0,
        time=0,
        dwExtraInfo=0,
    )

    up = INPUT()
    up.type = INPUT_KEYBOARD
    up.ki = KEYBDINPUT(
        wVk=VK_RETURN,
        wScan=scancode,
        dwFlags=KEYEVENTF_KEYUP,
        time=0,
        dwExtraInfo=0,
    )

    sent_down = user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
    if sent_down != 1:
        print(
            f"[bench] WARNING: SendInput(keydown) returned {sent_down}, err={ctypes.get_last_error()}",
            flush=True,
        )
        return False

    time.sleep(hold_sec)

    sent_up = user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))
    if sent_up != 1:
        print(
            f"[bench] WARNING: SendInput(keyup) returned {sent_up}, err={ctypes.get_last_error()}",
            flush=True,
        )
        return False
    return True


def kill_process_tree(pid: int) -> None:
    """Force-terminate process and all child processes."""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def kill_stale_victory() -> None:
    """Ensure no stray victory.exe process holds VRAM or file locks."""
    subprocess.run(
        ["taskkill", "/F", "/IM", "victory.exe", "/T"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def parse_frametimes_csv(path: Path) -> List[Tuple[int, int, int]]:
    """Return list of (frame, elapsed_us, frametime_us) rows from CSV."""
    rows: List[Tuple[int, int, int]] = []
    if not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("frame,"):
                    continue
                parts = line.split(",")
                if len(parts) != 3:
                    continue
                try:
                    rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def build_release() -> None:
    """Step 1: Build the release binary via `cargo build --release`."""
    print("[bench] Step 1: Running `cargo build --release`...", flush=True)
    t0 = time.monotonic()
    res = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if res.returncode != 0:
        print(f"[bench] ERROR: `cargo build --release` failed (rc={res.returncode}).", file=sys.stderr)
        sys.exit(1)
    print(f"[bench] Build succeeded in {time.monotonic() - t0:.1f}s.", flush=True)


def make_launch_env(game_dir: Path) -> dict:
    """Build deterministic environment matching `scripts/launch-game-fast.bat`."""
    env = os.environ.copy()
    defaults = {
        "VICTORY_GAME_DIR": str(game_dir),
        "VICTORY_CACHE_DIR": str(game_dir / "cache"),
        "VICTORY_DATA_DIR": str(game_dir / "data"),
        "VICTORY_CONFIG": str(REPO_ROOT / "victory.toml"),
        "VICTORY_UMA_BRIDGE": "1",
        "VICTORY_VELOCITY_IO": "1",
        "VICTORY_VELOCITY_IO_LEGACY_BCPACK": "1",
        "VICTORY_AMBIENT_AIR": "1.5",
        "VICTORY_TEXTURE_NO_WAIT": "1",
        "VICTORY_RELAY": "2",
        "VICTORY_WIDE_BACK": "1",
        "VICTORY_CARD_REHASH_RUNS": "1",
        "VICTORY_JOIN_COPIES": "1",
        "VICTORY_INDIRECT_WINDOW": "1",
        "VICTORY_RELAY_COPY": "1",
        "VICTORY_CARD_STEPS": "2",
        "VICTORY_INDIRECT_BATCH": "2",
        "VICTORY_MAP_THREADS": "1",
        "VICTORY_TITLE_PRIORITIES": "0",
        "VICTORY_CARD_AWAKE": "1",
        "VICTORY_SHOW_AFTER": "1",
        "VICTORY_AMBIENT": "0.8",
        "VICTORY_THIN": "1",
        "VICTORY_JOIN_PARTS": "8",
        "VICTORY_UPLOAD_AHEAD": "1",
        "VICTORY_LEVELS_ON_DEMAND": "1",
        "VICTORY_CARD_BUFFERS_MIN_KB": "64",
        "VICTORY_RAYTRACING": "0",
        "VICTORY_DLSS": "auto",
        "VICTORY_FRAMEGEN": "auto",
        "VICTORY_SHADOW_SCALE": "2",
        "VICTORY_DYNAMIC_SLOTS": "524288",
        "VICTORY_LOG": (
            "victory_gpu::host=info,"
            "victory_gpu::queue=info,"
            "victory_gpu::queue::behind=info,"
            "victory_gpu::shadow=info,"
            "victory_gpu::descriptor=info,"
            "victory_gpu::frame=info,"
            "victory_gpu::framegen=info,"
            "victory_gpu::dlss=info,"
            "victory_gpu::fsr=info,"
            "victory_hle::modules::dstorage=info,"
            "warn"
        ),
    }
    for k, v in defaults.items():
        env.setdefault(k, v)
    # Always force frametime telemetry output path for this benchmark run.
    env["VICTORY_FRAMETIMES"] = str(FRAMETIMES_CSV)
    # Also schedule Enter through victory-hle's GameInput bridge so menu skip
    # works reliably even if Windows UIPI blocks SendInput or focus shifts.
    env.setdefault("VICTORY_KEYS", "8:Enter,14:Enter,20:Enter,26:Enter,32:Enter")
    return env


def compute_metrics(dts_us: List[int]) -> dict:
    """Compute avg_fps, p1_low_fps, and frame-time statistics from microsecond deltas."""
    valid = [dt for dt in dts_us if dt > 0]
    frame_count = len(valid)
    if frame_count == 0:
        return {
            "frame_count": 0,
            "avg_fps": 0.0,
            "p1_low_fps": 0.0,
            "avg_ms": 0.0,
            "p1_low_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 0.0,
        }
    total_sec = sum(valid) / 1_000_000.0
    avg_fps = frame_count / total_sec if total_sec > 0 else 0.0
    avg_ms = (total_sec * 1000.0) / frame_count

    sorted_desc = sorted(valid, reverse=True)
    worst_n = max(1, math.ceil(frame_count * 0.01))
    p1_low_ms = (sum(sorted_desc[:worst_n]) / worst_n) / 1000.0
    p1_low_fps = 1000.0 / p1_low_ms if p1_low_ms > 0 else 0.0

    return {
        "frame_count": frame_count,
        "avg_fps": round(avg_fps, 2),
        "p1_low_fps": round(p1_low_fps, 2),
        "avg_ms": round(avg_ms, 2),
        "p1_low_ms": round(p1_low_ms, 2),
        "max_ms": round(sorted_desc[0] / 1000.0, 2),
        "min_ms": round(sorted_desc[-1] / 1000.0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VIctory automated FPS benchmark harness")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `cargo build --release` (only for debugging the harness).",
    )
    parser.add_argument(
        "--game-dir",
        type=Path,
        default=Path(os.environ.get("VICTORY_GAME_DIR", str(DEFAULT_GAME_DIR))),
        help="Path to the Xbox game dump directory.",
    )
    parser.add_argument(
        "--enter-delay",
        type=float,
        default=5.0,
        help="Seconds to wait after the game window appears and focuses before first Enter.",
    )
    parser.add_argument(
        "--extra-enter-at",
        type=float,
        nargs="*",
        default=[6.0, 12.0, 18.0, 24.0],
        help="Extra Enter presses at given seconds into the scene load wait to pass splash/menu screens.",
    )
    args = parser.parse_args()

    # Ensure no leftover victory.exe holds file locks before building.
    kill_stale_victory()

    # Step 1: Build
    if not args.skip_build:
        build_release()

    victory_exe = REPO_ROOT / "target" / "release" / "victory.exe"
    title_exe = args.game_dir / "game_scarlett_master.exe"
    if not victory_exe.is_file():
        print(f"[bench] ERROR: {victory_exe} not found.", file=sys.stderr)
        return 1
    if not title_exe.is_file():
        print(f"[bench] ERROR: {title_exe} not found.", file=sys.stderr)
        return 1

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in (FRAMETIMES_CSV, RUN_LOG):
        if old_file.exists():
            try:
                old_file.unlink()
            except OSError:
                pass

    env = make_launch_env(args.game_dir)
    cmd = [
        str(victory_exe),
        "run",
        str(title_exe),
        "--console",
        "series-s",
        "--execute",
    ]

    print(f"[bench] Step 2: Launching {' '.join(cmd)}", flush=True)
    run_start = time.monotonic()
    proc: Optional[subprocess.Popen] = None
    log_handle = RUN_LOG.open("w", encoding="utf-8", errors="replace")

    def check_alive_and_timeout(phase: str) -> None:
        elapsed = time.monotonic() - run_start
        if elapsed > HARD_TIMEOUT_SEC:
            raise TimeoutError(
                f"Hard timeout ({HARD_TIMEOUT_SEC:.0f}s) exceeded during {phase}"
            )
        assert proc is not None
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"Game process exited prematurely during {phase} (exit code {rc})"
            )

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        # Step 3: Auto-Input — wait for window to appear & gain focus, then send Enter
        print("[bench] Step 3: Waiting for game window to appear...", flush=True)
        hwnd: Optional[int] = None
        win_title = ""
        window_wait_limit = min(35.0, HARD_TIMEOUT_SEC - SCENE_LOAD_WAIT_SEC - MEASURE_WINDOW_SEC - 5.0)
        while time.monotonic() - run_start < window_wait_limit:
            check_alive_and_timeout("window wait")
            found = find_game_window(proc.pid)
            if found is not None:
                hwnd, win_title = found
                break
            time.sleep(0.25)

        if hwnd is None:
            raise TimeoutError("Game window did not appear before deadline")

        print(
            f"[bench] Window appeared at t={time.monotonic() - run_start:.1f}s "
            f"(hwnd={hwnd:#x}, title={win_title!r}). Focusing...",
            flush=True,
        )

        focused = False
        for _ in range(20):
            check_alive_and_timeout("focus wait")
            if focus_window(hwnd, proc.pid):
                focused = True
                break
            time.sleep(0.2)

        print(
            f"[bench] Window focus={focused}; waiting {args.enter_delay:.1f}s then sending Enter via SendInput...",
            flush=True,
        )
        time.sleep(args.enter_delay)
        focus_window(hwnd, proc.pid)
        if send_enter_key():
            print(f"[bench] Sent Enter via SendInput at t={time.monotonic() - run_start:.1f}s.", flush=True)
        else:
            print(
                f"[bench] SendInput(Enter) blocked by OS/UIPI at t={time.monotonic() - run_start:.1f}s "
                "(continuing with VICTORY_KEYS GameInput bridge).",
                flush=True,
            )

        # Step 4: Scene Load Wait (exact 120 seconds)
        print(
            f"[bench] Step 4: Waiting {SCENE_LOAD_WAIT_SEC:.0f}s for 3D scene load...",
            flush=True,
        )
        load_start = time.monotonic()
        extra_enters = sorted(t for t in args.extra_enter_at if 0.0 < t < SCENE_LOAD_WAIT_SEC)
        extra_idx = 0
        last_progress = 0.0
        while True:
            check_alive_and_timeout("scene load wait")
            load_elapsed = time.monotonic() - load_start
            if load_elapsed >= SCENE_LOAD_WAIT_SEC:
                break
            if extra_idx < len(extra_enters) and load_elapsed >= extra_enters[extra_idx]:
                focus_window(hwnd, proc.pid)
                send_enter_key()
                print(f"[bench] Sent extra Enter at load+{load_elapsed:.1f}s.", flush=True)
                extra_idx += 1
            if load_elapsed - last_progress >= 20.0:
                rows_now = parse_frametimes_csv(FRAMETIMES_CSV)
                print(
                    f"[bench]   load +{load_elapsed:.0f}s / {SCENE_LOAD_WAIT_SEC:.0f}s "
                    f"(total_presents={len(rows_now)})",
                    flush=True,
                )
                last_progress = load_elapsed
            time.sleep(min(0.5, SCENE_LOAD_WAIT_SEC - load_elapsed))

        # Step 5: Measure FPS over exact 20-second window (seconds 120-140)
        rows_before = parse_frametimes_csv(FRAMETIMES_CSV)
        start_frame_idx = rows_before[-1][0] if rows_before else 0
        print(
            f"[bench] Step 5: Sampling Present() frametimes for {MEASURE_WINDOW_SEC:.0f}s "
            f"(starting after frame #{start_frame_idx})...",
            flush=True,
        )
        measure_start = time.monotonic()
        while True:
            check_alive_and_timeout("measurement window")
            m_elapsed = time.monotonic() - measure_start
            if m_elapsed >= MEASURE_WINDOW_SEC:
                break
            time.sleep(min(0.25, MEASURE_WINDOW_SEC - m_elapsed))

        rows_after = parse_frametimes_csv(FRAMETIMES_CSV)
        window_rows = [r for r in rows_after if r[0] > start_frame_idx]
        dts_us = [r[2] for r in window_rows if r[2] > 0]
        metrics = compute_metrics(dts_us)

        if metrics["frame_count"] <= 0:
            raise RuntimeError(
                f"0 frames rendered during the 20s measurement window "
                f"(total frames before={len(rows_before)}, after={len(rows_after)})"
            )

        print("\n=== BENCHMARK RESULT ===", flush=True)
        print(f"avg_fps     : {metrics['avg_fps']:.2f}", flush=True)
        print(f"p1_low_fps  : {metrics['p1_low_fps']:.2f}", flush=True)
        print(f"frame_count : {metrics['frame_count']}", flush=True)
        print(
            f"frametime   : avg={metrics['avg_ms']:.2f}ms  "
            f"1%_low={metrics['p1_low_ms']:.2f}ms  "
            f"min={metrics['min_ms']:.2f}ms  max={metrics['max_ms']:.2f}ms",
            flush=True,
        )
        print(f"BENCH_JSON={json.dumps(metrics)}", flush=True)
        return 0

    except Exception as exc:
        print(f"[bench] FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        # Step 6: Clean Teardown
        if proc is not None:
            print(f"[bench] Step 6: Force-terminating PID {proc.pid} and child processes...", flush=True)
            kill_process_tree(proc.pid)
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        log_handle.close()
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
