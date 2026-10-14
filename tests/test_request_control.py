"""Request-local cancellation provenance and Windows idle-sleep lifecycle."""

import asyncio
import ctypes
import json
import signal
import sys
import threading
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import llm.cli
from llm.default_plugins import openwebui


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {"enabled_tool_ids": ["server:mcp:splunk-mcp"]},
    )
    monkeypatch.setattr(
        openwebui,
        "_client",
        lambda _: SimpleNamespace(
            base_url="https://example.test", token="synthetic", timeout=1200
        ),
    )
    monkeypatch.setattr(openwebui, "_get_model_item", lambda *args: {"id": "glm-5.3"})
    monkeypatch.setattr(openwebui, "_server_config", lambda _: {"version": "0.12.0"})
    monkeypatch.setattr(openwebui, "_enabled_knowledge_items", lambda *args: [])
    return openwebui.OpenWebUIModel("glm-5.3").conversation()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows desktop input regression")
def test_other_desktop_escape_does_not_cancel_generation(provider, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ctypes.windll.user32, "GetAsyncKeyState", lambda key: calls.append(key) or 1
    )

    async def runner(**kwargs):
        kwargs["on_text"]("finished")
        return {"answer": "finished", "native_checkpoint_count": 1}

    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", runner)
    try:
        assert provider.prompt("synthetic").text() == "finished"
    except KeyboardInterrupt:
        pytest.fail("Desktop-wide Escape incorrectly cancelled this response")
    assert calls == []


@pytest.mark.parametrize("interrupt", ["signal", "break", "unknown"])
def test_interrupted_chat_records_source_and_releases_worker(
    provider, monkeypatch, tmp_path, interrupt
):
    from llm import request_control

    if interrupt == "break" and not hasattr(signal, "SIGBREAK"):
        pytest.skip("Windows console break")

    calls = []
    released = threading.Event()

    class Kernel:
        def SetThreadExecutionState(self, flags):
            calls.append((flags, threading.get_ident()))
            if flags == 0x80000000:
                released.set()
            return 0x80000000

        def GetTickCount64(self):
            return 1000

        def QueryUnbiasedInterruptTime(self, pointer):
            pointer._obj.value = 9000000
            return 1

    monkeypatch.setattr(request_control, "_windows_kernel32", lambda: Kernel())
    prompts = iter(["synthetic", "quit"])
    monkeypatch.setattr(llm.cli, "_read_chat_prompt", lambda _: next(prompts))
    previous = signal.getsignal(signal.SIGINT)

    async def runner(**kwargs):
        kwargs["on_reasoning"]("in flight")
        await asyncio.to_thread(kwargs["stop_requested"].wait)
        raise openwebui.APIError("Chat request stopped by local client")

    def display(events, **kwargs):
        next(iter(events))
        if interrupt in ("signal", "break"):
            signal.raise_signal(
                signal.SIGINT if interrupt == "signal" else signal.SIGBREAK
            )
        else:
            raise KeyboardInterrupt()

    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", runner)
    monkeypatch.setattr(llm.cli, "display_stream_events", display)

    @click.command()
    def app():
        llm.cli._run_chat(
            provider.model.model_id,
            lambda prompt, *args: provider.prompt(prompt),
            conversation=provider,
        )

    result = CliRunner().invoke(app, catch_exceptions=False)
    released.wait()
    assert result.exit_code == 0
    assert signal.getsignal(signal.SIGINT) == previous
    records = [
        json.loads(line)
        for line in (tmp_path / f"{provider.id}.jsonl").read_text("utf-8").splitlines()
    ]
    cancellation = next(r for r in records if r.get("type") == "turn_cancelled")
    assert cancellation["source"] == (
        "process_signal"
        if interrupt in ("signal", "break")
        else "unknown_local_interrupt"
    )
    assert cancellation["signal"] == (
        {"signal": "SIGINT", "break": "SIGBREAK", "unknown": None}[interrupt]
    )
    assert "Cancellation source:" in result.output
    assert "Cancellation source:" in (tmp_path / f"{provider.id}.html").read_text(
        "utf-8"
    )
    assert any(
        r.get("type") == "provider_stop_requested"
        and r.get("reason") == "response_iterator_closed"
        for r in records
    )
    assert calls[0][0] == 0x80000001
    assert calls[-1] == (0x80000000, calls[0][1])
    assert calls[0][1] != threading.get_ident()


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("release_error", [False, True])
def test_power_guard_restores_prior_thread_state_even_on_error(
    monkeypatch, failure, enabled, release_error
):
    from llm import request_control

    flags = []
    records = []

    class Kernel:
        def SetThreadExecutionState(self, value):
            flags.append(value)
            if len(flags) > 1 and release_error:
                return 0
            return 0x80000002

        def GetTickCount64(self):
            return 5000

        def QueryUnbiasedInterruptTime(self, pointer):
            pointer._obj.value = 40000000
            return 1

    monkeypatch.setattr(request_control, "_windows_kernel32", lambda: Kernel())
    try:
        with request_control.prevent_idle_sleep(
            enabled=enabled, on_event=records.append
        ):
            if failure:
                raise ValueError("provider error")
    except ValueError:
        assert failure
    assert flags == ([0x80000001, 0x80000002] if enabled else [])
    assert records[0]["status"] == ("active" if enabled else "disabled")
    if enabled:
        assert records[-1]["status"] == (
            "release_failed" if release_error else "released"
        )
        assert records[-1]["power_sample"]["tick_count_ms"] == 5000


def test_power_failure_is_reported_without_interrupting_request(monkeypatch):
    from llm import request_control

    class Kernel:
        def SetThreadExecutionState(self, value):
            return 0

        def GetTickCount64(self):
            return 5000

        def QueryUnbiasedInterruptTime(self, pointer):
            return 0

    monkeypatch.setattr(request_control, "_windows_kernel32", lambda: Kernel())
    records = []
    with request_control.prevent_idle_sleep(enabled=True, on_event=records.append):
        records.append({"request": "continued"})
    assert records[0]["status"] == "failed"
    assert records[1] == {"request": "continued"}


def test_native_windows_power_request_is_acquired_and_released():
    from llm import request_control

    if sys.platform != "win32":
        pytest.skip("Windows API")
    records = []
    with request_control.prevent_idle_sleep(enabled=True, on_event=records.append):
        assert records[0]["status"] == "active"
    assert records[-1]["status"] == "released"
    assert records[-1]["power_sample"]["awake_time_100ns"] > 0


@pytest.mark.parametrize("outcome", ["complete", "error", "server_cancelled"])
@pytest.mark.parametrize("enabled", [False, True])
def test_provider_power_scope_and_server_failure_provenance(
    provider, monkeypatch, tmp_path, outcome, enabled
):
    from contextlib import contextmanager

    from llm.default_plugins.openwebui_socket import ServerCancellationError

    ownership = []

    @contextmanager
    def guard(*, enabled, on_event):
        ownership.append(("enter", enabled, threading.get_ident()))
        on_event({"status": "active" if enabled else "disabled"})
        try:
            yield
        finally:
            ownership.append(("exit", enabled, threading.get_ident()))
            on_event({"status": "released" if enabled else "disabled"})

    monkeypatch.setattr(openwebui, "prevent_idle_sleep", guard)

    async def runner(**kwargs):
        assert ownership[0][0] == "enter"
        if outcome == "server_cancelled":
            raise ServerCancellationError("synthetic policy stop")
        if outcome == "error":
            raise openwebui.APIError("synthetic provider failure")
        kwargs["on_text"]("finished")
        return {"answer": "finished", "native_checkpoint_count": 1}

    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", runner)
    prompts = iter(["synthetic", "quit"])
    monkeypatch.setattr(llm.cli, "_read_chat_prompt", lambda _: next(prompts))

    @click.command()
    def app():
        llm.cli._run_chat(
            provider.model.model_id,
            lambda prompt, *args: provider.prompt(prompt, openwebui_keep_awake=enabled),
            conversation=provider,
        )

    result = CliRunner().invoke(app, catch_exceptions=False)
    assert result.exit_code == 0
    assert ownership == [
        ("enter", enabled, ownership[0][2]),
        ("exit", enabled, ownership[0][2]),
    ]
    assert ownership[0][2] != threading.get_ident()
    records = [
        json.loads(line)
        for line in (tmp_path / f"{provider.id}.jsonl").read_text("utf-8").splitlines()
    ]
    assert not any(
        row.get("type") in ("provider_stop_requested", "turn_cancelled")
        for row in records
    )
    if outcome == "server_cancelled":
        error = next(row for row in records if row.get("type") == "turn_error")
        assert error["source"] == "server_event"
        assert error["server_event"] == "task-cancelled"
        assert error["server_reason"] == "synthetic policy stop"
    elif outcome == "complete":
        assert "finished" in result.output
        assert not any(row.get("type") == "turn_error" for row in records)
        assert records[-1]["type"] == "provider_power"
        assert records[-1]["status"] == ("released" if enabled else "disabled")


def test_non_interrupting_custom_signal_handler_is_preserved(monkeypatch):
    from llm.request_control import capture_interrupt_signals

    calls = []
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: calls.append(signum))
    try:
        with capture_interrupt_signals() as observed:
            signal.raise_signal(signal.SIGINT)
            assert observed == {}
        assert calls == [signal.SIGINT]
    finally:
        signal.signal(signal.SIGINT, previous)


def test_failed_native_power_request_does_not_fail_provider(
    provider, monkeypatch, tmp_path, capsys
):
    from llm import request_control

    def unavailable():
        raise OSError("synthetic API failure")

    monkeypatch.setattr(request_control, "_windows_kernel32", unavailable)

    async def runner(**kwargs):
        kwargs["on_text"]("finished")
        return {"answer": "finished", "native_checkpoint_count": 1}

    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", runner)
    assert provider.prompt("synthetic").text() == "finished"
    rows = [
        json.loads(line)
        for line in (tmp_path / f"{provider.id}.jsonl").read_text("utf-8").splitlines()
    ]
    assert any(
        row.get("type") == "provider_power" and row["status"] == "failed"
        for row in rows
    )
    assert "idle-sleep prevention failed" in capsys.readouterr().err
