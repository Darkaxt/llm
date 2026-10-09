"""Offline regression tests for durable native MCP checkpoints and recovery."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from llm.chat_journal import append_record, ensure_session, journal_path
from llm.default_plugins import openwebui, openwebui_socket
from llm.openwebui_recovery import (
    CheckpointError,
    build_resume_messages,
    load_recovery,
    save_tool_checkpoint,
)


@pytest.fixture
def recovery_journal(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    cid = "01m4ghy8nxzz29rq990znj4r5m"
    rid = cid + ":123456"
    ensure_session(cid, model="openwebui/glm-5.3")
    append_record(cid, {
        "type": "provider_request",
        "provider": "openwebui",
        "provider_run_id": rid,
        "model": "glm-5.3",
        "tool_ids": ["server:mcp:splunk-mcp"],
        "messages": [
            {"role": "system", "content": "TIDE configuration"},
            {"role": "user", "content": "Investigate the notable"},
        ],
    })
    return cid, rid, tmp_path


def _save_two_checkpoints(cid, rid):
    first = save_tool_checkpoint(cid, rid, {
        "call_id": "call-q1",
        "name": "splunk-mcp_splunk_run_query",
        "arguments": '{"query":"index=digit_sec | stats count"}',
        "output": {"results": [{"count": 2, "timestamp": 1791286979.676}]},
    })
    second = save_tool_checkpoint(cid, rid, {
        "call_id": "call-q2",
        "name": "splunk-mcp_splunk_run_query",
        "arguments": '{"query":"index=digit_sec | head 2"}',
        "output": [{"type": "output_text", "text": "Two raw AWS findings"}],
    })
    return first, second


def test_complete_checkpoint_sidecars_are_durable_and_replayable(recovery_journal):
    cid, rid, root = recovery_journal
    first, second = _save_two_checkpoints(cid, rid)
    assert first["type"] == "provider_tool_checkpoint"
    assert first["server_executed"] is True
    assert first["call_id"] == "call-q1"
    assert first["checkpoint_file"] != second["checkpoint_file"]
    sidecar = root / first["checkpoint_file"]
    assert sidecar.is_file()
    raw = sidecar.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == first["sha256"]
    assert json.loads(raw)["output"]["results"][0]["count"] == 2

    with pytest.raises(CheckpointError, match="no recorded provider error"):
        load_recovery(cid)

    append_record(cid, {
        "type": "provider_error",
        "provider": "openwebui",
        "provider_run_id": rid,
        "error": "Open WebUI: Server Connection Error",
    })
    recovered = load_recovery(cid)
    assert len(recovered["checkpoints"]) == 2
    assert recovered["checkpoints"][0]["name"] == "splunk-mcp_splunk_run_query"
    history = build_resume_messages(recovered)
    assert history[:2] == [
        {"role": "system", "content": "TIDE configuration"},
        {"role": "user", "content": "Investigate the notable"},
    ]
    assert history[2]["role"] == "assistant"
    assert history[2]["tool_calls"][0]["id"] == "call-q1"
    assert json.loads(history[2]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "index=digit_sec | stats count"
    }
    assert history[3]["role"] == "tool"
    assert history[3]["tool_call_id"] == "call-q1"
    assert json.loads(history[3]["content"])["results"][0]["count"] == 2
    assert history[4]["tool_calls"][0]["id"] == "call-q2"
    assert "No new tools are available" in history[-1]["content"]
    assert "Do not repeat completed searches" in history[-1]["content"]

    permit = build_resume_messages(recovered, allow_new_searches=True)
    assert "You may run NEW bounded" in permit[-1]["content"]
    assert permit[:-1] == history[:-1]


def test_checkpoint_repeated_result_is_idempotent(recovery_journal):
    cid, rid, _ = recovery_journal
    item = {
        "call_id": "repeat-q1",
        "name": "splunk-mcp_splunk_run_query",
        "arguments": '{"query":"index=digit_sec"}',
        "output": {"count": 3},
    }
    first = save_tool_checkpoint(cid, rid, item)
    second = save_tool_checkpoint(cid, rid, item)
    assert first["sha256"] == second["sha256"]
    item["output"] = {"count": 4}
    with pytest.raises(CheckpointError, match="Conflicting results"):
        save_tool_checkpoint(cid, rid, item)


def test_corrupted_checkpoint_fails_closed(recovery_journal):
    cid, rid, root = recovery_journal
    first, _ = _save_two_checkpoints(cid, rid)
    append_record(cid, {
        "type": "provider_error", "provider_run_id": rid, "error": "disconnected"
    })
    (root / first["checkpoint_file"]).write_text('{"tampered": true}')
    with pytest.raises(CheckpointError, match="checksum mismatch"):
        load_recovery(cid)


def test_missing_checkpoint_file_fails_closed(recovery_journal):
    cid, rid, root = recovery_journal
    first, _ = _save_two_checkpoints(cid, rid)
    append_record(cid, {
        "type": "provider_error", "provider_run_id": rid, "error": "disconnected"
    })
    (root / first["checkpoint_file"]).unlink()
    with pytest.raises(CheckpointError, match="sidecar unavailable"):
        load_recovery(cid)


def test_legacy_journal_with_only_tool_status_is_not_resumable(recovery_journal):
    cid, rid, _ = recovery_journal
    append_record(cid, {"type": "provider_tool", "tool": "↳ MCP done",
                        "provider_run_id": rid})
    append_record(cid, {"type": "provider_error", "provider_run_id": rid,
                        "error": "disconnected"})
    with pytest.raises(CheckpointError, match="Older journals"):
        load_recovery(cid)


def test_recovery_refuses_completed_turn(recovery_journal):
    cid, rid, _ = recovery_journal
    _save_two_checkpoints(cid, rid)
    append_record(cid, {"type": "provider_error", "provider_run_id": rid,
                        "error": "disconnected"})
    append_record(cid, {"type": "turn_completed", "turn_id": "completed"})
    with pytest.raises(CheckpointError, match="already completed"):
        load_recovery(cid)


def test_recovery_can_explicitly_assume_process_killed(recovery_journal):
    cid, rid, _ = recovery_journal
    _save_two_checkpoints(cid, rid)
    with pytest.raises(CheckpointError, match="no recorded provider error"):
        load_recovery(cid)
    assert len(load_recovery(cid, require_error=False)["checkpoints"]) == 2


def test_native_checkpoint_emitted_once_after_full_arguments_and_result():
    state = {
        "tool_done": {},
        "tool_results": [],
        "reasoning_blocks": [],
        "phase": "initial model",
        "phase_started_at": 0,
        "last_progress_at": 0,
        "stream_started": False,
    }
    checkpoints = []
    def handle(data):
        return openwebui_socket._consume_response_completion(
            data,
            state,
            on_text=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool=lambda _: None,
            on_status=lambda _: None,
            on_checkpoint=checkpoints.append,
        )

    handle({
        "type": "response.output_item.added",
        "item": {"type": "function_call", "call_id": "call-q1",
                 "name": "splunk-mcp_splunk_run_query", "arguments": ""},
    })
    handle({
        "type": "response.function_call_arguments.delta",
        "item_id": "call-q1",
        "delta": '{"query":"index=digit_sec"}',
    })
    handle({
        "type": "response.function_call_arguments.done",
        "item_id": "call-q1",
        "arguments": '{"query":"index=digit_sec"}',
    })
    handle({
        "type": "response.output_item.done",
        "item": {"type": "function_call_output", "call_id": "call-q1",
                 "output": [{"type": "output_text", "text": "2 events"}]},
    })
    assert len(checkpoints) == 1
    assert checkpoints[0]["call_id"] == "call-q1"
    assert json.loads(checkpoints[0]["arguments"])["query"] == "index=digit_sec"
    assert checkpoints[0]["output"] == [
        {"type": "output_text", "text": "2 events"}
    ]
    # Replayed snapshots must not duplicate checkpoints.
    openwebui_socket._capture_native_snapshot([
        {"type": "function_call", "call_id": "call-q1",
         "name": "splunk-mcp_splunk_run_query",
         "arguments": '{"query":"index=digit_sec"}'},
        {"type": "function_call_output", "call_id": "call-q1",
         "output": [{"type": "output_text", "text": "2 events"}]},
    ], state, checkpoints.append)
    assert len(checkpoints) == 1


def test_native_checkpoint_from_snapshot_survives_later_error(recovery_journal):
    cid, rid, _ = recovery_journal
    state = {}
    def checkpoint(item):
        save_tool_checkpoint(cid, rid, item)
    openwebui_socket._capture_native_snapshot([
        {
            "type": "function_call", "call_id": "call-q1",
            "name": "splunk-mcp_splunk_run_query",
            "arguments": '{"query":"index=digit_sec | stats count"}',
        },
        {
            "type": "function_call_output", "call_id": "call-q1",
            "output": {"results": [{"count": 2}]},
        },
    ], state, checkpoint)
    append_record(cid, {
        "type": "provider_error",
        "provider": "openwebui",
        "provider_run_id": rid,
        "error": "Server Connection Error after 300s",
    })
    recovered = load_recovery(cid)
    assert recovered["checkpoints"][0]["output"] == {"results": [{"count": 2}]}
    assert recovered["checkpoints"][0]["arguments"].startswith('{"query"')


def test_checkpoint_size_limit_fails_without_truncating(monkeypatch, recovery_journal):
    cid, rid, _ = recovery_journal
    from llm import openwebui_recovery
    monkeypatch.setattr(openwebui_recovery, "_CHECKPOINT_MAX_BYTES", 80)
    with pytest.raises(CheckpointError, match="above"):
        save_tool_checkpoint(cid, rid, {
            "call_id": "big",
            "name": "splunk-mcp_splunk_run_query",
            "arguments": '{"query":"test"}',
            "output": "A" * 500,
        })


def test_recovery_cli_dry_run_does_not_connect(monkeypatch, recovery_journal):
    cid, rid, _ = recovery_journal
    _save_two_checkpoints(cid, rid)
    append_record(cid, {
        "type": "provider_error", "provider_run_id": rid,
        "error": "server closed",
    })
    monkeypatch.setattr(openwebui, "_client", lambda *_: pytest.fail("client called"))
    @click.group()
    def app():
        pass
    openwebui.register_commands(app)
    output = CliRunner().invoke(app, ["openwebui", "resume", cid, "--dry-run"])
    assert output.exit_code == 0, output.output
    details = json.loads(output.output)
    assert details["checkpointed_calls"] == 2
    assert details["remote_task_resumed"] is False
    assert details["mode"] == "evidence-only"


@pytest.mark.parametrize("allow_new", [False, True])
def test_recovery_cli_uses_saved_evidence_without_replaying_queries(
    monkeypatch, recovery_journal, allow_new
):
    cid, rid, root = recovery_journal
    _save_two_checkpoints(cid, rid)
    append_record(cid, {
        "type": "provider_error", "provider_run_id": rid, "error": "timed out"
    })
    monkeypatch.setattr(openwebui, "_load_config", lambda: {
        "url": "https://example.test",
        "token": "test-jwt",
        "enabled_tool_ids": ["server:mcp:splunk-mcp", "server:mcp:other"],
    })
    monkeypatch.setattr(
        openwebui, "_client",
        lambda *_: SimpleNamespace(base_url="https://example.test",
                                   token="test-jwt", timeout=1200),
    )
    monkeypatch.setattr(
        openwebui, "_get_model_item",
        lambda client, model: {"id": model, "info": {"meta": {"capabilities": {
            "builtin_tools": True
        }}}},
    )
    monkeypatch.setattr(openwebui, "_guard_server_mcp_version",
                        lambda *args: ("0.11.3", True))
    calls = []
    async def fake_runner(**kwargs):
        calls.append(kwargs)
        kwargs["on_text"]("Recovered answer")
        return {"answer": "Recovered answer", "native_checkpoint_count": 0,
                "remote_chat_id": "temporary:server"}
    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", fake_runner)

    @click.group()
    def app():
        pass
    openwebui.register_commands(app)
    args = ["openwebui", "resume", cid]
    if allow_new:
        args.append("--allow-new-searches")
    output = CliRunner().invoke(app, args)
    assert output.exit_code == 0, output.output
    assert len(calls) == 1
    wire = calls[0]
    assert wire["messages"][0]["content"] == "TIDE configuration"
    assert wire["messages"][2]["tool_calls"][0]["id"] == "call-q1"
    assert wire["messages"][4]["tool_calls"][0]["id"] == "call-q2"
    if allow_new:
        assert wire["tool_ids"] == ["server:mcp:splunk-mcp"]
        assert "function_calling" not in wire["params"]
    else:
        assert wire["tool_ids"] == []
        assert wire["params"]["function_calling"] == "legacy"
    assert "Recovered answer" in output.output
    # A new journal is created, preserving the source journal untouched.
    source_records = [
        json.loads(line) for line in journal_path(cid).read_text().splitlines()
    ]
    recovered_id = next(
        rec["recovery_conversation_id"]
        for rec in source_records if rec.get("type") == "recovery_started"
    )
    new_records = [
        json.loads(line) for line in journal_path(recovered_id).read_text().splitlines()
    ]
    assert any(row.get("type") == "recovery_completed" for row in new_records)
    assert any(row.get("type") == "provider_request" for row in new_records)



def test_recovery_ignores_only_truncated_final_journal_line(recovery_journal):
    cid, rid, _ = recovery_journal
    _save_two_checkpoints(cid, rid)
    append_record(cid, {
        "type": "provider_error", "provider_run_id": rid,
        "error": "server timeout",
    })
    with journal_path(cid).open("ab") as handle:
        handle.write(b'{"type":"incomplete_tool')
    recovered = load_recovery(cid)
    assert len(recovered["checkpoints"]) == 2


def test_chat_journal_next_append_repairs_crash_tail(recovery_journal):
    cid, rid, _ = recovery_journal
    append_record(cid, {"type": "before_partial"})
    with journal_path(cid).open("ab") as handle:
        handle.write(b'{"interrupted"')
    append_record(cid, {"type": "after_partial"}, durable=True)
    rows = [json.loads(line) for line in journal_path(cid).read_text().splitlines()]
    assert rows[-2]["type"] == "before_partial"
    assert rows[-1]["type"] == "after_partial"
