"""Durable Open WebUI tool checkpoints and conservative, explicit recovery.

Each server-executed native tool result is written atomically before the model
continues. The JSONL journal contains a digest and a relative pointer, while
complete arguments and result payloads live in private local sidecar files.

A recovery is a NEW model request from saved evidence, not a reconnect to a
still-running Open WebUI task. It never re-executes saved tools by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .chat_journal import append_record, journal_dir, journal_path

_CHECKPOINT_MAX_BYTES = 32 * 1024 * 1024
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class CheckpointError(ValueError):
    """Invalid or unavailable checkpoint; do not silently restart searches."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise CheckpointError(
            f"Non-JSON tool evidence cannot be checkpointed: {exc}"
        ) from exc


def _safe_conversation_id(conversation_id: str) -> str:
    if not _ID_PATTERN.fullmatch(conversation_id):
        raise CheckpointError("Invalid conversation ID")
    return conversation_id


def _checkpoint_directory(conversation_id: str, provider_run_id: str) -> Path:
    _safe_conversation_id(conversation_id)
    run_hash = hashlib.sha256(provider_run_id.encode("utf-8")).hexdigest()[:24]
    return journal_dir() / (conversation_id + ".checkpoints") / run_hash


def save_tool_checkpoint(
    conversation_id: str,
    provider_run_id: str,
    checkpoint: dict[str, Any],
    *,
    inherited_from: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically persist the complete call+result before acknowledging it.

    A failed filesystem write raises to the event handler, stopping the
    investigation instead of leaving a silently uncheckpointed result.
    """
    if not isinstance(provider_run_id, str) or not provider_run_id:
        raise CheckpointError("Missing provider run ID")
    call_id = checkpoint.get("call_id")
    name = checkpoint.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise CheckpointError("Completed tool result has no call ID")
    if not isinstance(name, str) or not name or "output" not in checkpoint:
        raise CheckpointError("Completed tool result lacks tool name/output")
    if not isinstance(checkpoint.get("arguments"), (str, dict)):
        raise CheckpointError("Completed tool result lacks intact arguments")

    record = {
        "schema": "openwebui-native-tool-v1",
        "provider_run_id": provider_run_id,
        "call_id": call_id,
        "name": name,
        "arguments": checkpoint["arguments"],
        "output": checkpoint["output"],
        "completed_at": time.time(),
    }
    if inherited_from is not None:
        record["inherited_from"] = inherited_from
    data = _json_bytes(record)
    if len(data) > _CHECKPOINT_MAX_BYTES:
        raise CheckpointError(
            f"Tool result is {len(data)} bytes, above the "
            f"{_CHECKPOINT_MAX_BYTES}-byte checkpoint limit; "
            "refusing to discard evidence silently"
        )

    directory = _checkpoint_directory(conversation_id, provider_run_id)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    filename = hashlib.sha256(call_id.encode("utf-8")).hexdigest() + ".json"
    target = directory / filename
    digest = hashlib.sha256(data).hexdigest()

    # Idempotent when a final snapshot repeats a completed tool call.
    if target.exists():
        existing = target.read_bytes()
        if existing != data:
            # completed_at differs across repeated calls, but the payload must
            # remain identical. Compare the material call data, not timestamp.
            try:
                previous = json.loads(existing)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise CheckpointError("Existing checkpoint is corrupt") from exc
            for key in (
                "provider_run_id",
                "call_id",
                "name",
                "arguments",
                "output",
                "inherited_from",
            ):
                if previous.get(key) != record.get(key):
                    raise CheckpointError(
                        "Conflicting results for the same tool call ID"
                    )
        data = existing
        digest = hashlib.sha256(data).hexdigest()
    else:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=".writing-", delete=False
            ) as handle:
                temporary = handle.name
                try:
                    os.chmod(temporary, 0o600)
                except OSError:
                    pass
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    relative = target.relative_to(journal_dir()).as_posix()
    summary = {
        "type": "provider_tool_checkpoint",
        "provider": "openwebui",
        "provider_run_id": provider_run_id,
        "call_id": call_id,
        "tool": name,
        "checkpoint_file": relative,
        "sha256": digest,
        "size_bytes": len(data),
        "server_executed": True,
        "executed_in_this_run": inherited_from is None,
    }
    if inherited_from is not None:
        summary["inherited_from"] = inherited_from
    # Do not acknowledge a completed call until its journal pointer is durable.
    append_record(conversation_id, summary, durable=True)
    return summary


def _load_checkpoint(conversation_id: str, pointer: dict[str, Any]) -> dict[str, Any]:
    root = journal_dir().resolve()
    relative = pointer.get("checkpoint_file")
    if not isinstance(relative, str) or not relative:
        raise CheckpointError("Missing checkpoint sidecar path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise CheckpointError(f"Checkpoint sidecar unavailable: {relative}")
    raw = target.read_bytes()
    if len(raw) != pointer.get("size_bytes") or hashlib.sha256(
        raw
    ).hexdigest() != pointer.get("sha256"):
        raise CheckpointError(f"Checkpoint checksum mismatch: {relative}")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Checkpoint payload invalid: {relative}") from exc
    if (
        document.get("schema") != "openwebui-native-tool-v1"
        or document.get("provider_run_id") != pointer.get("provider_run_id")
        or document.get("call_id") != pointer.get("call_id")
        or document.get("name") != pointer.get("tool")
        or document.get("inherited_from") != pointer.get("inherited_from")
    ):
        raise CheckpointError("Checkpoint identity or format mismatch")
    if (
        not isinstance(document.get("arguments"), (str, dict))
        or "output" not in document
    ):
        raise CheckpointError("Incomplete tool checkpoint")
    return document


def load_recovery(
    conversation_id: str,
    *,
    provider_run_id: str | None = None,
    require_error: bool = True,
    allow_empty_answer: bool = False,
) -> dict[str, Any]:
    """Load verified checkpoints from the latest failed provider request.

    A completed turn is eligible only with an explicit empty-answer opt-in and
    a recorded empty answer. Later requests do not change the selected run.
    """
    _safe_conversation_id(conversation_id)
    path = journal_path(conversation_id)
    if not path.is_file():
        raise CheckpointError(f"No chat journal exists for {conversation_id}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                # A process can die halfway through the LAST append. All
                # previously fsynced checkpoint pointers remain usable.
                # Corruption in a newline-terminated/interior line is not
                # recoverable and must fail closed.
                if not line.endswith("\n") and handle.read(1) == "":
                    break
                raise CheckpointError(f"Malformed chat journal line {index}") from exc
            if isinstance(value, dict):
                records.append(value)

    requests = [
        r
        for r in records
        if r.get("type") == "provider_request"
        and r.get("provider") == "openwebui"
        and isinstance(r.get("provider_run_id"), str)
    ]
    if provider_run_id is not None:
        requests = [r for r in requests if r["provider_run_id"] == provider_run_id]
    if not requests:
        raise CheckpointError("No matching Open WebUI provider request in journal")
    request = requests[-1]
    run_id = request["provider_run_id"]
    start = next(i for i, row in enumerate(records) if row is request)
    end = next(
        (
            i
            for i in range(start + 1, len(records))
            if records[i].get("type") == "provider_request"
        ),
        len(records),
    )
    run_records = records[start:end]
    completed = [
        row
        for row in run_records
        if row.get("type") == "turn_completed"
        or (
            row.get("type") == "recovery_completed"
            and row.get("provider_run_id") == run_id
        )
    ]
    reason = "failed" if require_error else "assumed-stopped"
    if completed:
        if not allow_empty_answer:
            raise CheckpointError("The requested turn was already completed")
        terminal = completed[-1]
        answer = (
            (terminal.get("db_turn") or {}).get("response")
            if terminal["type"] == "turn_completed"
            else terminal.get("answer")
        )
        if not isinstance(answer, str) or answer.strip():
            raise CheckpointError("The completed answer is not demonstrably empty")
        reason = "empty-answer"
    elif require_error and not any(
        r.get("type") == "provider_error" and r.get("provider_run_id") == run_id
        for r in run_records
    ):
        raise CheckpointError(
            "The run has no recorded provider error. Do not resume an active "
            "or successfully completed request (use --assume-stopped only "
            "after confirming it has stopped)."
        )
    pointers = [
        r
        for r in run_records
        if r.get("type") == "provider_tool_checkpoint"
        and r.get("provider_run_id") == run_id
    ]
    if not pointers:
        raise CheckpointError(
            "No complete native tool checkpoints for this run. Older journals "
            "only captured tool status lines and cannot recreate missing results."
        )
    checkpoints: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for pointer in pointers:
        call_id = pointer.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise CheckpointError("Invalid checkpoint call ID")
        checkpoint = _load_checkpoint(conversation_id, pointer)
        if call_id in seen:
            if checkpoint != seen[call_id]:
                raise CheckpointError("Conflicting checkpoint records for call ID")
            continue
        checkpoints.append(checkpoint)
        seen[call_id] = checkpoint
    inherited = (request.get("recovery") or {}).get("inherited_checkpoint_count")
    if inherited is not None and inherited != sum(
        "inherited_from" in checkpoint for checkpoint in checkpoints
    ):
        raise CheckpointError("Incomplete inherited checkpoint set")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise CheckpointError("The original request has no replayable message history")
    if not isinstance(request.get("model"), str) or not request["model"]:
        raise CheckpointError("Missing original model")
    return {
        "conversation_id": conversation_id,
        "provider_run_id": run_id,
        "request": request,
        "checkpoints": checkpoints,
        "recovery_reason": reason,
    }


def build_resume_messages(
    recovery: dict[str, Any], *, allow_new_searches: bool = False
) -> list[dict[str, Any]]:
    """Reconstruct valid assistant tool-call/tool-output history without re-calling tools."""
    original = recovery["request"]["messages"]
    # Round-trip to deep-copy the persisted JSON messages; never mutate journal data.
    messages: list[dict[str, Any]] = json.loads(json.dumps(original))
    checkpoints = recovery["checkpoints"]
    if not checkpoints:
        raise CheckpointError("Recovery requires at least one completed tool result")
    for item in checkpoints:
        args = item["arguments"]
        arguments = (
            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        )
        try:
            parsed = json.loads(arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"Tool {item['call_id']} arguments cannot be replayed safely"
            ) from exc
        if not isinstance(parsed, dict):
            raise CheckpointError("Tool arguments are not a JSON object")
        name = item["name"]
        call_id = item["call_id"]
        output = item["output"]
        content = (
            output
            if isinstance(output, str)
            else json.dumps(output, ensure_ascii=False)
        )
        # A failed recovery's original request already contains inherited pairs.
        # Verify them against its own sidecars instead of appending duplicates.
        prior_calls = [
            call
            for message in messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", [])
            if call.get("id") == call_id
        ]
        prior_results = [
            message
            for message in messages
            if message.get("role") == "tool" and message.get("tool_call_id") == call_id
        ]
        if prior_calls or prior_results:
            if (
                len(prior_calls) != 1
                or len(prior_results) != 1
                or prior_calls[0].get("function")
                != {"name": name, "arguments": arguments}
                or prior_results[0].get("content") != content
            ):
                raise CheckpointError("Conflicting tool checkpoint and message history")
            continue
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }
        )
    mode = (
        "You may run NEW bounded read-only searches if essential, but must not "
        "repeat the completed tool calls supplied in the transcript. "
        if allow_new_searches
        else "No new tools are available. Base the report solely on the preserved "
        "tool results and identify all unresolved questions explicitly. "
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Continue the interrupted investigation using the exact completed "
                "tool-call results above. These results were durably checkpointed "
                "during the previous attempt. Do not assume the "
                "old remote task is still running. Do not repeat completed searches. "
                + mode
                + "Separate verified observations from hypotheses, retain source "
                "scope and timestamps, and provide a concise evidence ledger."
            ),
        }
    )
    return messages
