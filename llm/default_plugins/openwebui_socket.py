# Vendored compatibility shim from vedmaka/openwebui-sdk sockets.py
# Source commit: f55e6391173d46bb9d664ab7129fde8b57c40497
# Local change: allow Open WebUI top-level files metadata in tool-enabled chats.

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
import json
import time
import uuid
from asyncio import TimeoutError as _AsyncTimeoutError
from collections.abc import Callable
from typing import Any

from openwebui_sdk import http
from openwebui_sdk.errors import APIError, ConfigError
from openwebui_sdk.render import (
    extract_all_reasoning,
    extract_answer_text,
    extract_tool_events,
    format_tool_line,
)


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
        if isinstance(item, dict) and item.get("type") == "message":
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


async def run_chat_with_tools_with_files(
    *,
    base_url: str,
    token: str,
    model: str,
    messages: list[dict[str, str]],
    tool_ids: list[str],
    files: list[dict[str, Any]] | None = None,
    timeout: int = 300,
    on_text: Callable[[str], None] | None = None,
    on_tool: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """Run a tool-enabled chat over the Socket.IO path.

    Returns a structured result: ``{answer, reasoning, tool_calls}`` (reasoning
    and tool_calls are None/empty when the model didn't produce them). Streaming
    callers wire the ``on_*`` callbacks to print incrementally and discard the
    return value; ``ask --json`` ignores the callbacks and just reads the result.
    """
    socketio = _need_socketio()
    import aiohttp  # validated present by _need_socketio()

    if not tool_ids and not chat_id:
        raise ConfigError(
            "run_chat_with_tools requires either tool_ids (tool execution) or "
            "a chat_id (chat persistence via --save)"
        )

    out_text = on_text or (lambda _s: None)
    out_tool = on_tool or (lambda _s: None)
    out_status = on_status or (lambda _s: None)
    out_reasoning = on_reasoning or (lambda _s: None)

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

    state: dict[str, Any] = {
        "answer": "",        # last rendered answer prose (stripped)
        "raw_content": "",   # serialized legacy content or current structured output
        "reasoning_blocks": [],  # list of reasoning strings
        "tool_done": {},     # tool key -> done? (to detect executing->done)
        "tool_results": [],  # [{name, result}] for structured result capture
        "error": None,
        "last_event_at": time.monotonic(),
        "phase": "initial model",
        "phase_started_at": time.monotonic(),
    }
    done = asyncio.Event()
    remote_task_ids: list[str] = []
    remote_chat_id: str | None = None

    def _emit_answer_snapshot(answer: str) -> None:
        if answer.startswith(state["answer"]):
            delta = answer[len(state["answer"]):]
            if delta:
                out_text(delta)
            state["answer"] = answer
        elif len(answer) > len(state["answer"]):
            state["answer"] = answer

    def _sync_structured_output(output: Any) -> None:
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
            elif block.startswith(prev_blocks[i]):
                delta = block[len(prev_blocks[i]):]
                if delta:
                    out_reasoning(delta)
        state["reasoning_blocks"] = current_blocks

        current_events = _structured_tool_events(output)
        new_results: list[dict[str, Any]] = []
        for ev in current_events:
            key = ev.get("call_id") or ev["name"]
            done_now = bool(ev.get("done"))
            done_before = state["tool_done"].get(key)
            if done_before is None or done_before != done_now:
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
            state["tool_results"] = new_results

    def _handle_event(payload: dict[str, Any]) -> None:
        # Current Open WebUI emits on Socket.IO "events"; older releases used
        # "chat-events". Both carry {chat_id, message_id, data}.
        if payload.get("chat_id") != chat_id:
            return
        state["last_event_at"] = time.monotonic()

        event = payload.get("data") or {}
        etype = event.get("type")
        data = event.get("data") or {}

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
            return

        if etype in ("chat:message", "replace"):
            content = str(data.get("content") or "")
            if content:
                _emit_answer_snapshot(content)
            return

        if etype == "response:completion":
            # These are provider-native Responses API stream events. Current
            # Open WebUI also emits periodic/final chat:completion snapshots,
            # which are authoritative and easier to consume here.
            return

        if etype != "chat:completion":
            return

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
                elif block.startswith(prev_blocks[i]):
                    delta = block[len(prev_blocks[i]):]
                    if delta:
                        out_reasoning(delta)
            state["reasoning_blocks"] = current_blocks

            current_tool_events = extract_tool_events(content)
            results_by_name = {r["name"]: r for r in state["tool_results"]}
            for ev in current_tool_events:
                name = ev.get("name") or "tool"
                done_now = bool(ev.get("done"))
                done_before = state["tool_done"].get(name)
                if done_before is None or done_before != done_now:
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
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "chat_id": chat_id,
            "id": message_id,
            "session_id": session_id,
            "tool_ids": tool_ids,
            "files": files or None,
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
            },
            "variables": {},
        }
        # NOTE: we intentionally do NOT send params.function_calling here. The
        # server reads it from the model config (model_info.params.function_calling,
        # main.py:1131) - a model with native FC configured uses it automatically,
        # one without uses Open WebUI's prompt-based calling. The CLI never
        # overrides model config.

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

        # ---- wait for completion, with visible idle heartbeat ----
        started_at = time.monotonic()
        heartbeat_seconds = 15.0
        while not done.is_set():
            elapsed = time.monotonic() - started_at
            remaining = timeout - elapsed
            if remaining <= 0:
                state["error"] = f"timed out after {timeout}s"
                break
            try:
                await asyncio.wait_for(
                    done.wait(),
                    timeout=min(heartbeat_seconds, remaining),
                )
            except _AsyncTimeoutError:
                now = time.monotonic()
                idle = now - state["last_event_at"]
                phase_elapsed = now - state["phase_started_at"]
                # Keep the socket/session alive exactly as the browser does.
                with contextlib.suppress(Exception):
                    await sio.emit("heartbeat", {})
                out_status(
                    f"waiting for {state['phase']} "
                    f"({int(phase_elapsed)}s phase, {int(elapsed + heartbeat_seconds)}s total, "
                    f"{int(idle)}s since last event)"
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
    }
