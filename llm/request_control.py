"""Process-signal provenance and request-scoped Windows idle-sleep prevention."""

from __future__ import annotations

import ctypes
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def capture_interrupt_signals() -> Iterator[dict[str, Any]]:
    """Observe this process's console signals, restoring existing handlers."""
    observed: dict[str, Any] = {}
    previous: dict[int, Any] = {}

    def interrupted(signum, frame):
        handler = previous[signum]
        try:
            if callable(handler):
                handler(signum, frame)
            elif handler == signal.SIG_DFL:
                signal.default_int_handler(signum, frame)
        except KeyboardInterrupt:
            observed.update(source="process_signal", signal=signal.Signals(signum).name)
            raise

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
                if signum is None:
                    continue
                handler = signal.getsignal(signum)
                if handler == signal.SIG_IGN:
                    continue
                previous[signum] = handler
                signal.signal(signum, interrupted)
        yield observed
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _windows_kernel32():
    if sys.platform != "win32":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint32
    kernel.GetTickCount64.argtypes = []
    kernel.GetTickCount64.restype = ctypes.c_uint64
    kernel.QueryUnbiasedInterruptTime.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    kernel.QueryUnbiasedInterruptTime.restype = ctypes.c_int
    return kernel


def windows_power_sample() -> dict[str, Any]:
    """Raw sleep-inclusive/awake-only clocks; never infer a lock or its cause."""
    try:
        kernel = _windows_kernel32()
        if kernel is None:
            return {}
        awake = ctypes.c_uint64()
        sample: dict[str, Any] = {"tick_count_ms": kernel.GetTickCount64()}
        if kernel.QueryUnbiasedInterruptTime(ctypes.byref(awake)):
            sample["awake_time_100ns"] = awake.value
        else:
            sample["awake_time_100ns"] = None
        return sample
    except (OSError, AttributeError) as exc:
        return {"unavailable": type(exc).__name__}


def local_interrupt_details(observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": observed.get("source", "unknown_local_interrupt"),
        "signal": observed.get("signal"),
        "reason": "KeyboardInterrupt",
        "power_sample": windows_power_sample(),
    }


@contextmanager
def prevent_idle_sleep(
    *, enabled: bool, on_event: Callable[[dict[str, Any]], None]
) -> Iterator[None]:
    """Acquire and restore execution state on the SAME active request thread.

    Only idle system sleep is requested off. Display, locking, explicit sleep
    and global power configuration remain under Windows/user policy.
    """
    kernel = None
    previous = 0
    record: dict[str, Any] = {"status": "disabled"}
    try:
        if enabled:
            kernel = _windows_kernel32()
            if kernel is None:
                record = {"status": "unsupported"}
            else:
                previous = kernel.SetThreadExecutionState(
                    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
                )
                record = {"status": "active" if previous else "failed"}
                if not previous:
                    record["error_code"] = getattr(
                        ctypes, "get_last_error", lambda: None
                    )()
    except (OSError, AttributeError) as exc:
        record = {"status": "failed", "error_class": type(exc).__name__}
    try:
        on_event({**record, "power_sample": windows_power_sample()})
        yield
    finally:
        if previous and kernel is not None:
            try:
                restored = kernel.SetThreadExecutionState(previous | _ES_CONTINUOUS)
                record = {"status": "released" if restored else "release_failed"}
                if not restored:
                    record["error_code"] = getattr(
                        ctypes, "get_last_error", lambda: None
                    )()
            except (OSError, AttributeError) as exc:
                record = {"status": "release_failed", "error_class": type(exc).__name__}
            on_event({**record, "power_sample": windows_power_sample()})
