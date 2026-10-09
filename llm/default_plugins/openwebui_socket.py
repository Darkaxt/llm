# Vendored compatibility shim from vedmaka/openwebui-sdk sockets.py
# Source commit: f55e6391173d46bb9d664ab7129fde8b57c40497
# Local change: allow Open WebUI top-level files metadata in tool-enabled chats.
# Recovery: preserve a live Socket.IO subscription after sessionless HTTP loss.

"""Socket.IO chat runner - the only path over which Open WebUI actually
*executes* tools.

Why this module exists
----------------------
Open WebUI 0.6.5 executes the tool-call loop only when a chat request carries
``session_id`` + ``chat_id`` + ``message_id`` AND the client holds a Socket.IO
session to receive the results (see ``backend/open_webui/utils/middleware.py``:
``event_emitter`` is set only with all three ids, and the tool-execution loop
lives in ``post_response_handler``, which emits content via Socket.IO
``chat-events``). The plain HTTP streaming path clients otherwise get does no
tool execution and no round-2 re-prompt.

So when the caller wants tools, this module:

1. connects a ``python-socketio`` async client, authenticated with the bearer
   token (``backend/open_webui/socket/main.py:connect`` reads ``auth['token']``)
2. POSTs ``/api/chat/completions`` with ``session_id`` = the socket sid, plus a
   fresh ``chat_id``/``id`` and ``tool_ids`` -> the server returns a tiny ack
   ``{"status": true, "task_id": ...}`` and runs the real work in a background
   task, delivering content over the socket
3. listens for ``chat-events`` filtered to our ``chat_id``, streams the answer
   text to ``on_text``, tool activity to ``on_tool`` and status to ``on_status``
4. returns when it sees the terminating ``chat:completion`` event with
   ``data.done == true`` (or an error / cancellation)

``python-socketio`` (and its aiohttp async HTTP layer) are required runtime
dependencies of openwebui-cli; they are imported lazily so a missing install
fails with a clear message rather than at module import time.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import threading
import time
import uuid
from asyncio import TimeoutError as _AsyncTimeoutError
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from openwebui_sdk import http
from openwebui_sdk.errors import APIError, ConfigError
from openwebui_sdk.render import (
    extract_all_reasoning,
    extract_answer_text,
    extract_tool_events,
    format_tool_line,
)


def _sessionless_recovery_grace() -> float:
    """How long to keep listening for a final server event after HTTP disconnects."""
    value = os.environ.get("LLM_OPENWEBUI_RECOVERY_GRACE", "60")
    try:
        seconds = float(value)
    except ValueError as exc:
        raise ConfigError("LLM_OPENWEBUI_RECOVERY_GRACE must be a number of seconds") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ConfigError("LLM_OPENWEBUI_RECOVERY_GRACE must be non-negative and finite")
    return min(seconds, 600.0)


async def _recover_sessionless_completion(
    done: asyncio.Event,
    state: dict[str, Any],
    error: Exception,
    on_status: Callable[[str], None],
    *,
    grace_seconds: float,
) -> None:
    """Do not replay a potentially executing MCP request after HTTP loss.

    Open WebUI can still deliver its final chat event through Socket.IO after
    the proxy has closed the synchronous HTTP response. Prefer that result over
    a duplicate request, which could execute remote tools twice.
    """
    if done.is_set() or state.get("error"):
        return
    on_status(
        "HTTP connection lost; waiting up to "
        f"{grace_seconds:g}s for the original server response"
    )
    if grace_seconds:
        try:
            await asyncio.wait_for(done.wait(), timeout=grace_seconds)
        except _AsyncTimeoutError:
            pass
    if not done.is_set() and not state.get("error"):
        state["error"] = (
            f"sessionless Open WebUI HTTP connection lost: {error}. "
            "No final server event was received; the remote request may still "
            "be running. Use !retry in chat to explicitly resend the failed "
            "turn (read-only tools only)."
        )


def _meaningful_progress_timeout() -> float:
    """Timeout for real model/tool activity, independent of socket heartbeats."""
    value = os.environ.get("LLM_OPENWEBUI_PROGRESS_TIMEOUT", "1200")
    try:
        seconds = float(value)
    except ValueError as exc:
        raise ConfigError(
            "LLM_OPENWEBUI_PROGRESS_TIMEOUT must be a number of seconds"
        ) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ConfigError(
            "LLM_OPENWEBUI_PROGRESS_TIMEOUT must be positive and finite"
        )
    return seconds


def _stalled_request_error(
    state: dict[str, Any],
    now: float,
    *,
    event_timeout: float,
    progress_timeout: float,
) -> str | None:
    """Socket keepalives cannot extend the deadline for useful work."""
    no_progress = now - float(state["last_progress_at"])
    if no_progress >= progress_timeout:
        return (
            f"Open WebUI stalled: no model/tool progress for {int(no_progress)}s "
            f"during {state['phase']} (limit {progress_timeout:g}s). "
            "Set LLM_OPENWEBUI_PROGRESS_TIMEOUT to adjust the limit."
        )
    no_event = now - float(state["last_event_at"])
    if no_event >= event_timeout:
        return f"timed out after {event_timeout:g}s without an Open WebUI event"
    return None


async def _stop_remote_chat_tasks(
    session: Any,
    *,
    base_url: str,
    token: str,
    chat_id: str,
    on_status: Callable[[str], None],
) -> bool:
    """Cancel only the caller's chat tasks via Open WebUI's user-scoped API."""
    endpoint = (
        f"{base_url.rstrip('/')}/api/tasks/chat/{quote(chat_id, safe='')}/stop"
    )
    try:
        async with session.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        ) as response:
            if response.status >= 400:
                on_status(f"could not stop remote chat tasks (HTTP {response.status})")
                return False
            on_status("remote chat task cancellation requested")
            return True
    except Exception as exc:
        on_status(f"could not stop remote chat tasks: {exc}")
        return False


def _need_socketio() -> Any:
    """Import python-socketio (and aiohttp), raising a clear ConfigError if either
    is missing.

    Both are required runtime dependencies of openwebui-cli (tool execution
    routes through Open WebUI's Socket.IO path). python-socketio's AsyncClient
    needs aiohttp for the Engine.IO HTTP handshake (even when the websocket
    transport is selected, the initial handshake is HTTP). Without it the
    connect fails with the cryptic ``aiohttp not installed -- cannot make HTTP
    requests!``. We catch a missing import here with an actionable reinstall
    message rather than letting engineio's internals surface.
    """
    try:
        import socketio  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised at runtime only
        raise ConfigError(
            "tool support needs 'python-socketio', a required dependency; "
            "reinstall with: pip install -e openwebui-cli"
        ) from exc
    try:
        import aiohttp  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised at runtime only
        raise ConfigError(
            "tool support needs 'aiohttp' (python-socketio's async HTTP layer), "
            "a required dependency; reinstall with: pip install -e openwebui-cli"
        ) from exc
    return socketio


def _structured_parts_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if text is not None:
            chunks.append(str(text))
    return "".join(chunks)


def _structured_output_text(output: Any) -> str:
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role", "assistant") == "assistant"
        ):
            text = _structured_parts_text(item.get("content"))
            if text.strip():
                chunks.append(text)
    return "\n".join(chunks)


def _structured_reasoning(output: Any) -> list[str]:
    if not isinstance(output, list):
        return []
    blocks: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        parts = item.get("summary") if isinstance(item.get("summary"), list) else item.get("content")
        text = _structured_parts_text(parts)
        if text:
            blocks.append(text)
    return blocks


def _structured_tool_events(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, list):
        return []
    results_by_call: dict[str, Any] = {}
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        call_id = str(item.get("call_id") or "")
        raw = item.get("output")
        if isinstance(raw, list):
            result = _structured_parts_text(raw)
        elif raw is None:
            result = None
        else:
            result = raw
        results_by_call[call_id] = result

    events: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = str(item.get("call_id") or item.get("id") or "")
        status = str(item.get("status") or "")
        done = call_id in results_by_call or status in {"failed", "incomplete"}
        events.append(
            {
                "name": str(item.get("name") or "tool"),
                "call_id": call_id,
                "arguments": item.get("arguments"),
                "result": results_by_call.get(call_id),
                "done": done,
            }
        )
    return events



def _capture_server_tool_sources(
    sources: Any,
    state: dict[str, Any],
    *,
    tool_ids: list[str],
    on_tool: Callable[[str], None],
    on_source: Callable[[dict[str, Any]], None],
) -> int:
    """Record legacy tool execution evidence emitted by Open WebUI.

    With function_calling=legacy, Open WebUI runs its preliminary tool-call
    handler before starting the assistant model stream. It returns results in
    chat:completion data.sources (tool_result=True), not in the native
    function_call output items. Source IDs follow get_source_context()'s
    ordered, unique metadata.source mapping.
    """
    if not isinstance(sources, list):
        return 0
    captured = 0
    source_ids = state.setdefault("source_ids", {})
    seen = state.setdefault("server_tool_source_keys", set())
    server_ids = [
        item[len("server:mcp:") :]
        for item in tool_ids
        if item.startswith("server:mcp:")
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        origin = source.get("source")
        origin = origin if isinstance(origin, dict) else {}
        metadata = source.get("metadata")
        metadata = metadata if isinstance(metadata, list) else []
        # Assign IDs for *all* sources (including non-tools) to match the
        # IDs the model receives from Open WebUI's get_source_context().
        for meta in metadata:
            if not isinstance(meta, dict):
                continue
            source_key = str(
                meta.get("source") or origin.get("id") or "N/A"
            )
            if source_key not in source_ids:
                source_ids[source_key] = len(source_ids) + 1
        if source.get("tool_result") is not True:
            continue

        name = str(origin.get("name") or "")
        if not name:
            continue
        tool_metadata = next(
            (item for item in metadata if isinstance(item, dict)), {}
        )
        source_key = str(
            tool_metadata.get("source") or origin.get("id") or "N/A"
        )
        citation_id = source_ids.setdefault(
            source_key, len(source_ids) + 1
        )
        documents = source.get("document")
        content = "\n".join(str(doc) for doc in documents) if isinstance(documents, list) else ""
        params = tool_metadata.get("parameters")
        params = params if isinstance(params, dict) else {}
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        dedup = (
            name,
            digest,
            json.dumps(params, sort_keys=True, default=str),
        )
        if dedup in seen:
            continue
        seen.add(dedup)

        is_mcp = any(
            name == server_id or name.startswith(server_id + "_")
            for server_id in server_ids
        )
        record: dict[str, Any] = {
            "source_id": citation_id,
            "tool": name,
            "is_mcp": is_mcp,
            "server_executed": True,
            "parameters": params,
            "result_chars": len(content),
            "result_sha256": digest,
            # Bounded evidence for post-incident audit without allowing a
            # large Splunk result to bloat the local JSONL journal.
            "result_excerpt": content[:32768],
            "result_truncated": len(content) > 32768,
        }
        state.setdefault("server_tool_sources", []).append(record)
        # Legacy calls have no native function_call event; expose their
        # result in the same structured response as native tool calls.
        state.setdefault("tool_results", []).append(
            {
                "name": name,
                "result": content[:32768],
                "server_executed": True,
                "source_id": citation_id,
            }
        )
        state["last_progress_at"] = time.monotonic()
        on_source(record)
        label = "MCP" if is_mcp else "tool"
        on_tool(
            f"↳ {label} {name} returned (source [{citation_id}], "
            f"{len(content)} chars)"
        )
        captured += 1
    return captured


def _reconcile_answer_snapshot(
    answer: str,
    state: dict[str, Any],
    on_text: Callable[[str], None],
) -> None:
    """Append only genuinely new text from an authoritative full snapshot."""
    previous = state["answer"]
    if answer.startswith(previous):
        suffix = answer[len(previous):]
        if suffix:
            on_text(suffix)
            state["last_progress_at"] = time.monotonic()
        state["answer"] = answer
    elif len(answer) > len(previous):
        # A revised/normalized server snapshot cannot safely rewrite terminal
        # output that has already streamed. Store its canonical final value
        # without printing a duplicate or garbled suffix.
        state["answer"] = answer
        state["last_progress_at"] = time.monotonic()



def _record_native_tool_item(
    item: dict[str, Any],
    state: dict[str, Any],
    *,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Join native tool-call and tool-result items by call ID.

    The server emits both event-level items and cumulative chat:completion
    snapshots. Only emit a complete call+result pair, and only once.
    """
    kind = item.get("type")
    call_id = str(item.get("call_id") or item.get("id") or "")
    if not call_id:
        return
    if kind == "function_call":
        old = state.setdefault("native_call_items", {}).get(call_id) or {}
        # An output_item.added snapshot can contain empty arguments even
        # after argument deltas have populated the same call ID. Never erase
        # a complete argument string with an empty/stale placeholder.
        fields = {
            key: value for key, value in item.items()
            if value is not None and (value != "" or not old.get(key))
        }
        state["native_call_items"][call_id] = {**old, **fields}
    elif kind == "function_call_output" and "output" in item:
        state.setdefault("native_result_items", {})[call_id] = item
    else:
        return
    _emit_native_checkpoint_if_ready(state, call_id, on_checkpoint)


def _emit_native_checkpoint_if_ready(
    state: dict[str, Any],
    call_id: str,
    on_checkpoint: Callable[[dict[str, Any]], None] | None,
) -> None:
    if on_checkpoint is None:
        return
    emitted = state.setdefault("native_checkpoint_ids", set())
    if call_id in emitted:
        return
    call = state.get("native_call_items", {}).get(call_id)
    result = state.get("native_result_items", {}).get(call_id)
    if not isinstance(call, dict) or not isinstance(result, dict):
        return
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or not name:
        return
    if not isinstance(arguments, (dict, str)) or not arguments:
        # Wait for response.function_call_arguments.done or a full snapshot.
        return
    # A partial argument string must never become a "complete" checkpoint.
    try:
        decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(decoded, dict):
        return
    checkpoint = {
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "output": result["output"],
    }
    # Callback must finish its disk write before we mark this call recorded.
    on_checkpoint(checkpoint)
    emitted.add(call_id)


def _capture_native_snapshot(
    output: Any,
    state: dict[str, Any],
    on_checkpoint: Callable[[dict[str, Any]], None] | None,
) -> None:
    if not isinstance(output, list):
        return
    # Record all calls first, because snapshots may put outputs after calls
    # from an earlier generation or contain a reordered prior_output segment.
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            _record_native_tool_item(item, state, on_checkpoint=on_checkpoint)
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            _record_native_tool_item(item, state, on_checkpoint=on_checkpoint)


def _consume_response_completion(
    data: dict[str, Any],
    state: dict[str, Any],
    *,
    on_text: Callable[[str], None],
    on_reasoning: Callable[[str], None],
    on_tool: Callable[[str], None],
    on_status: Callable[[str], None],
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> bool:
    """Forward live Open WebUI Responses events without waiting for snapshots.

    Open WebUI 0.11.x converts Chat Completions and Responses API streams to
    Socket.IO response:completion. chat:completion carries reconciliation
    snapshots and final messages, not per-token updates.
    """
    kind = str(data.get("type") or "")
    is_answer = kind == "response.output_text.delta"
    is_reasoning = kind in {
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
    }
    if is_answer or is_reasoning:
        delta = data.get("delta")
        if not isinstance(delta, str) or not delta:
            return False
        if not state.get("stream_started"):
            state["stream_started"] = True
            on_status("live model token stream started")
        if is_answer:
            on_text(delta)
            state["answer"] += delta
            phase = "model streaming"
        else:
            # Update the same reasoning state read by final-snapshot handling,
            # preventing the final full reasoning from being shown twice.
            key = (data.get("item_id"), data.get("output_index"), kind)
            keys = state.setdefault("reasoning_stream_keys", [])
            blocks = state["reasoning_blocks"]
            if key not in keys:
                keys.append(key)
                if blocks:
                    on_reasoning("\n\n")
                blocks.append("")
            blocks[keys.index(key)] += delta
            on_reasoning(delta)
            phase = "model reasoning"
        if state.get("phase") != phase:
            state["phase"] = phase
            state["phase_started_at"] = time.monotonic()
        state["last_progress_at"] = time.monotonic()
        state["stream_delta_count"] = state.get("stream_delta_count", 0) + 1
        return True

    if kind in {"response.output_item.added", "response.output_item.done"}:
        item = data.get("item")
        if not isinstance(item, dict):
            return False
        item_type = item.get("type")
        if item_type in {"function_call", "function_call_output"}:
            _record_native_tool_item(
                item, state, on_checkpoint=on_checkpoint
            )
        if item_type == "function_call":
            name = str(item.get("name") or "")
            call_id = str(
                item.get("call_id") or item.get("id")
                or f"output:{data.get('output_index')}"
            )
            if name and call_id not in state["tool_done"]:
                state["tool_done"][call_id] = False
                state.setdefault("tool_names", {})[call_id] = name
                on_tool(f"↳ {name} ...")
                state["last_progress_at"] = time.monotonic()
            # Finishing an argument item is not tool execution completing.
            if kind.endswith(".done"):
                state["phase"] = "tool execution"
                state["phase_started_at"] = time.monotonic()
            return True
        if item_type == "function_call_output":
            call_id = str(item.get("call_id") or "")
            if call_id and not state["tool_done"].get(call_id):
                name = state.get("tool_names", {}).get(call_id, "tool")
                state["tool_done"][call_id] = True
                state["tool_results"].append(
                    {"name": name, "result": item.get("output")}
                )
                on_tool(f"↳ {name} done")
                on_status("tool result received; waiting for model continuation")
                state["phase"] = "model continuation"
                state["phase_started_at"] = time.monotonic()
                state["last_progress_at"] = time.monotonic()
            return True
        return False

    if kind in {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    }:
        call_id = str(data.get("item_id") or data.get("call_id") or "")
        if call_id:
            entry = state.setdefault("native_call_items", {}).setdefault(
                call_id, {"type": "function_call", "call_id": call_id}
            )
            if kind.endswith(".done"):
                arguments = data.get("arguments")
                if isinstance(arguments, str):
                    entry["arguments"] = arguments
            elif isinstance(data.get("delta"), str):
                entry["arguments"] = (
                    str(entry.get("arguments") or "") + data["delta"]
                )
            _emit_native_checkpoint_if_ready(state, call_id, on_checkpoint)
        # Do not leak complete tool arguments to the terminal status area.
        # Generating tool arguments still counts as meaningful progress.
        state["last_progress_at"] = time.monotonic()
        if state.get("phase") != "preparing tool call":
            state["phase"] = "preparing tool call"
            state["phase_started_at"] = time.monotonic()
        return True

    return False


def _build_browser_chat_body(
    *,
    model: str,
    model_item: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    tool_ids: list[str],
    files: list[dict[str, Any]] | None,
    params: dict[str, Any] | None,
    chat_id: str,
    session_id: str | None,
    message_id: str,
    user_message_id: str,
) -> dict[str, Any]:
    """Build an Open WebUI 0.11.3 compatible Socket.IO chat payload.

    The website's new-chat request omits chat_id and messages so the server
    persists history itself; the CLI deliberately supplies a temporary chat_id
    and explicit messages so its pre-resolved local KB context is preserved.
    """
    last_user_content = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                last_user_content = content
            else:
                last_user_content = json.dumps(content, ensure_ascii=False)
            break

    body = {
        "model": model,
        "model_item": model_item or {"id": model},
        # The browser's new-chat request leaves messages absent and lets the
        # server build conversation history. We retain the explicit messages
        # here because the CLI injects the pre-resolved TIDE KB context and
        # manages its own local conversation state. This is deliberate.
        "messages": messages,
        "stream": True,
        "params": dict(params or {}),
        # v0.11.3 browser sends message_ids for model fan-out. Keep id as
        # backward-compatible fallback for older Open WebUI deployments.
        "chat_id": chat_id,
        "id": message_id,
        "message_ids": [
            {"model_id": model, "message_id": message_id, "modelIdx": 0}
        ],
        "parent_id": None,
        "user_message": {
            "id": user_message_id,
            "parentId": None,
            "childrenIds": [message_id],
            "role": "user",
            "content": last_user_content,
            "timestamp": int(time.time()),
            "models": [model],
        },
        "tool_ids": tool_ids,
        "tool_servers": [],
        "files": files or [],
        "features": {
            "image_generation": False,
            "code_interpreter": False,
            "web_search": False,
        },
        "variables": {},
        "chat_variables": {},
        "background_tasks": {},
    }
    if session_id:
        body["session_id"] = session_id
    return body


async def run_chat_with_tools_with_files(
    *,
    base_url: str,
    token: str,
    model: str,
    model_item: dict[str, Any] | None = None,
    messages: list[dict[str, Any]],
    tool_ids: list[str],
    files: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 300,
    on_text: Callable[[str], None] | None = None,
    on_tool: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    on_source: Callable[[dict[str, Any]], None] | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    chat_id: str | None = None,
    sessionless_server_tools: bool = False,
    stop_requested: threading.Event | None = None,
) -> dict[str, Any]:
    """Run a tool-enabled chat over the Socket.IO path.

    Returns a structured result: ``{answer, reasoning, tool_calls}`` (reasoning
    and tool_calls are None/empty when the model didn't produce them). Streaming
    callers wire the ``on_*`` callbacks to print incrementally and discard the
    return value; ``ask --json`` ignores the callbacks and just reads the result.
    """
    socketio = _need_socketio()
    import aiohttp  # validated present by _need_socketio()

    # An evidence-only recovery deliberately has no tools and uses a temporary
    # chat ID allocated after the Socket.IO handshake. Do not require a caller-
    # supplied chat ID or enable tools just to get an answer.
    if not isinstance(messages, list) or not messages:
        raise ConfigError("Open WebUI chat requires a non-empty message history")

    out_text = on_text or (lambda _s: None)
    out_tool = on_tool or (lambda _s: None)
    out_status = on_status or (lambda _s: None)
    out_reasoning = on_reasoning or (lambda _s: None)
    out_source = on_source or (lambda _record: None)
    out_checkpoint = on_checkpoint

    # python-socketio's aiohttp session defaults to trust_env=False, so it would
    # IGNORE HTTP_PROXY/HTTPS_PROXY and bypass the vault proxy that injects
    # credentials for hosts like https://open-web-ui.teq.wiki - the gateway then
    # returns 400 at the Engine.IO handshake. The core CLI (urllib) honors these
    # env vars automatically; we make socketio do the same by giving it an
    # aiohttp ClientSession with trust_env=True. (http_session is a constructor
    # kwarg on AsyncClient, NOT on connect() - it was removed from connect in
    # python-socketio 5.x.)
    http_session = aiohttp.ClientSession(trust_env=True)
    sio = socketio.AsyncClient(http_session=http_session)
    # ``chat_id`` may be caller-supplied for a persisted chat. When it is
    # omitted, defer choosing it until after Socket.IO connects: current Open
    # WebUI treats a bare UUID as a persisted chat ID and returns 404 if that
    # chat does not exist. Unsaved browser chats use "temporary:<socket-id>".
    message_id = str(uuid.uuid4())
    user_message_id = str(uuid.uuid4())

    state: dict[str, Any] = {
        "answer": "",        # last rendered answer prose (stripped)
        "raw_content": "",   # serialized legacy content or current structured output
        "reasoning_blocks": [],  # list of reasoning strings
        "tool_done": {},     # tool key -> done? (to detect executing->done)
        "tool_results": [],  # [{name, result}] for structured result capture
        "error": None,
        "last_event_at": time.monotonic(),
        "last_progress_at": time.monotonic(),
        "phase": "initial model",
        "phase_started_at": time.monotonic(),
        "stream_started": False,
        "stream_delta_count": 0,
        "server_tool_sources": [],
    }
    done = asyncio.Event()
    remote_task_ids: list[str] = []
    remote_chat_id: str | None = None

    def _emit_answer_snapshot(answer: str) -> None:
        _reconcile_answer_snapshot(answer, state, out_text)

    def _sync_structured_output(output: Any) -> None:
        _capture_native_snapshot(output, state, out_checkpoint)
        answer = _structured_output_text(output)
        if answer:
            _emit_answer_snapshot(answer)

        current_blocks = _structured_reasoning(output)
        prev_blocks = state["reasoning_blocks"]
        for i, block in enumerate(current_blocks):
            if i >= len(prev_blocks):
                if i > 0:
                    out_reasoning("\n\n")
                out_reasoning(block)
                state["last_progress_at"] = time.monotonic()
            elif block.startswith(prev_blocks[i]):
                delta = block[len(prev_blocks[i]):]
                if delta:
                    out_reasoning(delta)
                    state["last_progress_at"] = time.monotonic()
        state["reasoning_blocks"] = current_blocks

        current_events = _structured_tool_events(output)
        new_results: list[dict[str, Any]] = []
        for ev in current_events:
            key = ev.get("call_id") or ev["name"]
            done_now = bool(ev.get("done"))
            done_before = state["tool_done"].get(key)
            if done_before is None or done_before != done_now:
                state["last_progress_at"] = time.monotonic()
                suffix = " done" if done_now else " ..."
                out_tool(f"↳ {ev['name']}{suffix}")
                if done_now:
                    state["phase"] = "model continuation"
                    state["phase_started_at"] = time.monotonic()
                    out_status("tool result received; waiting for model continuation")
                else:
                    state["phase"] = "tool execution"
                    state["phase_started_at"] = time.monotonic()
            state["tool_done"][key] = done_now
            if done_now:
                new_results.append(
                    {
                        "name": ev["name"],
                        "result": ev.get("result"),
                    }
                )
        if new_results:
            existing = {
                (str(item.get("name")), str(item.get("source_id", "")))
                for item in state["tool_results"]
            }
            for result in new_results:
                key = (str(result.get("name")), "")
                if key not in existing:
                    state["tool_results"].append(result)
                    existing.add(key)

    def _handle_event(payload: dict[str, Any]) -> None:
        # Current Open WebUI emits on Socket.IO "events"; older releases used
        # "chat-events". Both carry {chat_id, message_id, data}.
        if payload.get("chat_id") != chat_id:
            return
        state["last_event_at"] = time.monotonic()

        event = payload.get("data") or {}
        etype = event.get("type")
        data = event.get("data") or {}
        # Socket.IO status/heartbeat/duplicate snapshot events are not progress.

        if etype == "source":
            _capture_server_tool_sources(
                [data],
                state,
                tool_ids=tool_ids,
                on_tool=out_tool,
                on_source=out_source,
            )
            return

        if etype == "status":
            action = data.get("action") or data.get("description")
            if action:
                out_status(str(action))
            return

        if etype in ("task-cancelled", "chat:tasks:cancel"):
            # chat:tasks:cancel follows a normal completion in current Open WebUI,
            # so only treat the explicit task-cancelled event as an error.
            if etype == "task-cancelled":
                state["error"] = "task cancelled by server"
                done.set()
            return

        if etype == "chat:message:error":
            err = data.get("error")
            if isinstance(err, dict):
                state["error"] = err.get("content") or err.get("detail") or str(err)
            else:
                state["error"] = str(err or "Open WebUI chat error")
            done.set()
            return

        if etype in ("chat:message:delta", "message"):
            fragment = data.get("content") or ""
            if fragment:
                out_text(str(fragment))
                state["answer"] += str(fragment)
                state["last_progress_at"] = time.monotonic()
            return

        if etype in ("chat:message", "replace"):
            content = str(data.get("content") or "")
            if content:
                _emit_answer_snapshot(content)
            return

        if etype in ("response:completion", "chat:completion"):
            # Open WebUI emits live token and argument deltas here.
            # Some backends wrap Responses events in chat:completion.
            if isinstance(data, dict) and str(data.get("type") or "").startswith("response."):
                _consume_response_completion(
                    data,
                    state,
                    on_text=out_text,
                    on_reasoning=out_reasoning,
                    on_tool=out_tool,
                    on_status=out_status,
                    on_checkpoint=out_checkpoint,
                )
                return
            if etype == "response:completion":
                return

        if etype != "chat:completion":
            return

        if isinstance(data, dict) and "sources" in data:
            # OWUI 0.11.x emits source evidence BEFORE streaming tokens.
            # In legacy FC this is the only visible record of server-side
            # MCP execution; final output.function_call may be empty.
            count = _capture_server_tool_sources(
                data["sources"],
                state,
                tool_ids=tool_ids,
                on_tool=out_tool,
                on_source=out_source,
            )
            if count:
                out_status(f"{count} server-side tool result(s) verified")

        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            if isinstance(err, dict):
                state["error"] = (
                    err.get("detail")
                    or err.get("content")
                    or err.get("message")
                    or str(err)
                )
            else:
                state["error"] = str(err)
            done.set()
            return

        output = data.get("output")
        if isinstance(output, list):
            _sync_structured_output(output)
            state["raw_content"] = json.dumps(output, ensure_ascii=False)
        else:
            content = data.get("content") or ""

            # Legacy serialized-content path.
            current_blocks = extract_all_reasoning(content)
            prev_blocks = state["reasoning_blocks"]
            for i, block in enumerate(current_blocks):
                if i >= len(prev_blocks):
                    if i > 0:
                        out_reasoning("\n\n")
                    out_reasoning(block)
                    state["last_progress_at"] = time.monotonic()
                elif block.startswith(prev_blocks[i]):
                    delta = block[len(prev_blocks[i]):]
                    if delta:
                        out_reasoning(delta)
                        state["last_progress_at"] = time.monotonic()
            state["reasoning_blocks"] = current_blocks

            current_tool_events = extract_tool_events(content)
            results_by_name = {r["name"]: r for r in state["tool_results"]}
            for ev in current_tool_events:
                name = ev.get("name") or "tool"
                done_now = bool(ev.get("done"))
                done_before = state["tool_done"].get(name)
                if done_before is None or done_before != done_now:
                    state["last_progress_at"] = time.monotonic()
                    out_tool(format_tool_line(ev))
                    if done_now:
                        state["phase"] = "model continuation"
                        state["phase_started_at"] = time.monotonic()
                        out_status("tool result received; waiting for model continuation")
                    else:
                        state["phase"] = "tool execution"
                        state["phase_started_at"] = time.monotonic()
                state["tool_done"][name] = done_now

                if done_now:
                    if name in results_by_name:
                        results_by_name[name]["result"] = ev.get("result")
                        results_by_name[name]["done"] = True
                    else:
                        entry = {
                            "name": name,
                            "result": ev.get("result"),
                            "done": True,
                        }
                        results_by_name[name] = entry
                        state["tool_results"].append(entry)

            answer = extract_answer_text(content)
            if answer:
                _emit_answer_snapshot(answer)
            state["raw_content"] = content

        if data.get("done"):
            if not state["stream_started"] and state["answer"]:
                out_status(
                    "final answer received without intermediate token deltas; "
                    "check upstream streaming support"
                )
            done.set()

    async def _on_events(payload, cb=None):  # type: ignore[no-untyped-def]
        try:
            _handle_event(payload)
        except Exception as exc:  # noqa: BLE001
            state["error"] = f"error handling chat event: {exc}"
            done.set()
        if cb:
            with contextlib.suppress(Exception):
                await cb(True)

    # Current Open WebUI uses "events"; retain the old name for 0.6.x
    # deployments supported by openwebui-sdk.
    sio.on("events", handler=_on_events)
    sio.on("chat-events", handler=_on_events)

    # ---- connect ----
    # Open WebUI mounts its socket.io ASGI app at "/ws" with
    # socketio_path="/ws/socket.io" (backend/open_webui/main.py:937 +
    # socket/main.py:155); the frontend uses the same path (+layout.svelte:65).
    #
    # Transport order matters: the server registers EITHER ["websocket"] (when
    # ENABLE_WEBSOCKET_SUPPORT=True, the default) OR ["polling"] - it does NOT
    # accept both (see socket/main.py: transports=(...
    #   ["websocket"] if ENABLE_WEBSOCKET_SUPPORT else ["polling"])). So a client
    # that offers ["polling","websocket"] probes polling first and gets a
    # 400 "Invalid transport". We put websocket first (the default server
    # config) and keep polling as a fallback for servers with websockets off.
    connect_kwargs: dict[str, Any] = {
        "auth": {"token": token},
        "transports": ["websocket", "polling"],
        "socketio_path": "/ws/socket.io",
    }
    out_status("connecting to Open WebUI tool session")
    try:
        await sio.connect(base_url, **connect_kwargs)
    except Exception as exc:
        # Close the proxy-aware session and any half-open engineio session to
        # avoid "Unclosed client session" warnings on exit.
        with contextlib.suppress(Exception):
            await sio.disconnect()
        with contextlib.suppress(Exception):
            await http_session.close()
        raise APIError(f"failed to connect socket.io session: {exc}") from exc

    # Open WebUI routes chat events to the Socket.IO namespace session. The
    # client's ``sid`` attribute is the lower-level Engine.IO session and can
    # differ from the namespace SID returned in the Socket.IO connect packet.
    # Sending the Engine.IO SID starts the chat but routes its events nowhere.
    session_id = sio.get_sid("/")
    if not session_id:
        raise APIError("socket.io connection has no default namespace session id")

    out_status("tool session connected")

    # Mirror the browser's post-connect authentication handshake. The server's
    # connect(auth=...) path should already populate SESSION_POOL and join the
    # user:<id> room, but user-join is what the current frontend explicitly
    # performs after every connection. Requiring its ACK proves that this
    # socket can actually receive the user's `events` broadcasts.
    try:
        joined_user = await sio.call(
            "user-join",
            {"auth": {"token": token}},
            timeout=min(float(timeout), 30.0),
        )
    except Exception as exc:
        raise APIError(f"Open WebUI user-join handshake failed: {exc}") from exc
    if not isinstance(joined_user, dict) or not joined_user.get("id"):
        raise APIError(
            f"Open WebUI user-join returned an invalid response: {joined_user!r}"
        )
    out_status(f"socket authenticated as {joined_user.get('name') or joined_user['id']}")

    if not chat_id:
        chat_id = f"temporary:{session_id}"

    # Everything from here on is wrapped so the socket session and the proxy-aware
    # aiohttp session are ALWAYS torn down - no "Unclosed client session" warnings
    # on any exit path (POST failure, bad ack, timeout, success).
    try:
        # ---- POST the chat (server returns an ack; content comes over socket) ----
        # Mirror the frontend payload shape (Chat.svelte) - in particular
        # ``features`` MUST be a dict, NOT omitted: Open WebUI's
        # post_response_handler does ``metadata.get("features", {}).get(...)``
        # (middleware.py:1544), and metadata["features"] defaults to None when
        # omitted (main.py:1124). So a minimal client that omits it makes the
        # background task crash with AttributeError('NoneType' ... 'get') before
        # emitting any chat-events - the request 200s but you get nothing.
        request_session_id = (
            None if sessionless_server_tools else session_id
        )
        body = _build_browser_chat_body(
            model=model,
            model_item=model_item,
            messages=messages,
            tool_ids=tool_ids,
            files=files,
            params=params,
            chat_id=chat_id,
            session_id=request_session_id,
            message_id=message_id,
            user_message_id=user_message_id,
        )
        # Native mode keeps the model's default function-calling behavior.
        # Background legacy mode explicitly opts out of hidden builtin tools
        # while retaining the selected server-side MCP tools.

        if sessionless_server_tools:
            # Current Open WebUI only injects hidden builtin tools for requests
            # carrying session_id. Keep this authenticated socket connected to
            # the user room for events, but omit session_id from the POST so the
            # server resolves only explicit tool_ids (e.g. Splunk MCP). Server-
            # side MCP functions still execute in the native streaming tool loop.
            remote_chat_id = chat_id
            out_status(
                "request submitted · native server tools · hidden builtins suppressed"
            )
            async def post_sessionless_chat() -> Any:
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                request_timeout = aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=min(float(timeout), 60.0),
                    sock_read=None,
                )
                async with http_session.post(
                    f"{base_url}/api/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=request_timeout,
                ) as response:
                    raw = await response.text()
                    if response.status >= 400:
                        detail: Any = raw
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                detail = (
                                    parsed.get("detail")
                                    or parsed.get("message")
                                    or parsed.get("error")
                                    or parsed
                                )
                            else:
                                detail = parsed
                        except Exception:
                            pass
                        raise APIError(
                            "Open WebUI sessionless chat request failed "
                            f"({response.status}): {detail}"
                        )
                    if not raw.strip():
                        return None
                    try:
                        return json.loads(raw)
                    except Exception:
                        return raw

            post_task = asyncio.create_task(post_sessionless_chat())

            started_at = time.monotonic()
            state["last_progress_at"] = started_at
            state["last_event_at"] = started_at
            heartbeat_seconds = 15.0
            last_heartbeat = started_at
            progress_limit = _meaningful_progress_timeout()
            while not done.is_set() and not post_task.done():
                now = time.monotonic()
                if stop_requested is not None and stop_requested.is_set():
                    state["error"] = "Chat request cancelled by user"
                    post_task.cancel()
                    break
                problem = _stalled_request_error(
                    state, now, event_timeout=timeout, progress_timeout=progress_limit
                )
                if problem:
                    state["error"] = problem
                    post_task.cancel()
                    break
                remaining_idle = timeout - (now - state["last_event_at"])
                remaining_progress = progress_limit - (
                    now - state["last_progress_at"]
                )
                wait_for = min(1.0, remaining_idle, remaining_progress)
                try:
                    await asyncio.wait_for(done.wait(), timeout=wait_for)
                except _AsyncTimeoutError:
                    now = time.monotonic()
                    if now - last_heartbeat < heartbeat_seconds:
                        continue
                    elapsed = now - started_at
                    idle = now - state["last_event_at"]
                    phase_elapsed = now - state["phase_started_at"]
                    last_heartbeat = now
                    with contextlib.suppress(Exception):
                        await sio.emit("heartbeat", {})
                    out_status(
                        f"waiting for {state['phase']} "
                        f"({int(phase_elapsed)}s phase, {int(elapsed)}s total, "
                        f"{int(idle)}s since last event, "
                        f"{int(now - state['last_progress_at'])}s since progress)"
                    )

            try:
                await post_task
            except asyncio.CancelledError:
                if not state["error"]:
                    state["error"] = "sessionless Open WebUI request cancelled"
            except Exception as exc:
                if not state["error"]:
                    if isinstance(exc, (aiohttp.ClientError, OSError, _AsyncTimeoutError)):
                        await _recover_sessionless_completion(
                            done,
                            state,
                            exc,
                            out_status,
                            grace_seconds=_sessionless_recovery_grace(),
                        )
                    else:
                        state["error"] = (
                            f"sessionless Open WebUI request failed: {exc}"
                        )

            if not done.is_set() and not state["error"]:
                state["error"] = (
                    "Open WebUI request completed without a final chat event"
                )
        else:
            ack = await asyncio.to_thread(
                http.json_request,
                f"{base_url}/api/chat/completions",
                method="POST",
                token=token,
                json_body=body,
                timeout=timeout,
                endpoint="POST /api/chat/completions (socket session)",
            )

            if not (isinstance(ack, dict) and ack.get("status")):
                raise APIError(
                    f"unexpected chat ack: {ack!r}; server did not start a background task"
                )

            task_ids = ack.get("task_ids")
            if not isinstance(task_ids, list):
                task_id = ack.get("task_id")
                task_ids = [task_id] if task_id else []
            remote_task_ids = [str(task_id) for task_id in task_ids if task_id]
            ack_chat_id = ack.get("chat_id")
            remote_chat_id = str(ack_chat_id) if ack_chat_id else chat_id
            details = []
            if remote_task_ids:
                details.append("task " + ",".join(remote_task_ids))
            if remote_chat_id:
                details.append(f"chat {remote_chat_id}")
            out_status(
                "request accepted" + (f" ({'; '.join(details)})" if details else "")
            )
            state["last_progress_at"] = time.monotonic()
            state["last_event_at"] = state["last_progress_at"]

        # ---- wait for completion, with visible idle heartbeat ----
        #
        # `timeout` is an inactivity timeout, not a total wall-clock limit.
        # Long investigations can legitimately run for many minutes while
        # emitting retrieval/tool/model events. Resetting the deadline on every
        # real event prevents active investigations being killed at 600s.
        started_at = time.monotonic()
        heartbeat_seconds = 15.0
        last_heartbeat = started_at
        progress_limit = _meaningful_progress_timeout()
        while not sessionless_server_tools and not done.is_set():
            now = time.monotonic()
            if stop_requested is not None and stop_requested.is_set():
                state["error"] = "Chat request cancelled by user"
            else:
                state["error"] = _stalled_request_error(
                    state, now, event_timeout=timeout, progress_timeout=progress_limit
                )
            if state["error"]:
                if remote_task_ids:
                    await _stop_remote_chat_tasks(
                        http_session,
                        base_url=base_url,
                        token=token,
                        chat_id=remote_chat_id or chat_id,
                        on_status=out_status,
                    )
                break
            remaining_idle = timeout - (now - state["last_event_at"])
            remaining_progress = progress_limit - (now - state["last_progress_at"])
            wait_for = min(1.0, remaining_idle, remaining_progress)
            try:
                await asyncio.wait_for(done.wait(), timeout=wait_for)
            except _AsyncTimeoutError:
                now = time.monotonic()
                if now - last_heartbeat < heartbeat_seconds:
                    continue
                elapsed = now - started_at
                idle = now - state["last_event_at"]
                phase_elapsed = now - state["phase_started_at"]
                last_heartbeat = now
                # Keep the socket/session alive exactly as the browser does.
                with contextlib.suppress(Exception):
                    await sio.emit("heartbeat", {})
                out_status(
                    f"waiting for {state['phase']} "
                    f"({int(phase_elapsed)}s phase, {int(elapsed)}s total, "
                    f"{int(idle)}s since last event, "
                    f"{int(now - state['last_progress_at'])}s since progress)"
                )
    finally:
        with contextlib.suppress(Exception):
            await sio.disconnect()
        with contextlib.suppress(Exception):
            await http_session.close()

    if state["error"]:
        raise APIError(str(state["error"]))
    # Return a structured result so callers (notably ``ask --json``) can capture
    # reasoning + tool calls alongside the answer. The streaming callers that
    # print via callbacks discard this value.
    return {
        "answer": state["answer"],
        "reasoning": "\n\n".join(state["reasoning_blocks"]) or None,
        "tool_calls": [
            {"name": r.get("name"), "result": r.get("result")}
            for r in state["tool_results"]
        ],
        # Full serialized content with <details> reasoning/tool_calls blocks —
        # needed by --save so the web UI can render them.
        "raw_content": state["raw_content"],
        "remote_chat_id": remote_chat_id or chat_id,
        "remote_task_ids": remote_task_ids,
        "server_tool_sources": state["server_tool_sources"],
        "native_checkpoint_count": len(state.get("native_checkpoint_ids", ())),
    }
