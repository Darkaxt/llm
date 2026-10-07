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
        "raw_content": "",   # last full serialized content (with <details> blocks)
        "reasoning_blocks": [],  # list of reasoning strings (one per <details type=reasoning>)
        "tool_done": {},     # tool name -> done? (to detect executing->done)
        "tool_results": [],  # [{name, result}] for --json capture
        "error": None,
    }
    done = asyncio.Event()

    def _handle_event(payload: dict[str, Any]) -> None:
        # Server emits {chat_id, message_id, data}; only ours.
        if payload.get("chat_id") != chat_id:
            return
        event = payload.get("data") or {}
        etype = event.get("type")
        data = event.get("data") or {}

        if etype == "status":
            action = data.get("action") or data.get("description")
            if action:
                out_status(str(action))
            return

        if etype == "task-cancelled":
            state["error"] = "task cancelled by server"
            done.set()
            return

        if etype != "chat:completion":
            return

        # Error embedded in a completion event.
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            state["error"] = err.get("detail") if isinstance(err, dict) else str(err)
            done.set()
            return

        content = data.get("content") or ""

        # Surface ALL reasoning blocks (there can be multiple in a multi-step
        # tool-calling turn — the model reasons before each tool call). Each
        # block is streamed via prefix-diff as it grows; NEW blocks (index beyond
        # what we've seen) are printed in full.
        current_blocks = extract_all_reasoning(content)
        prev_blocks = state["reasoning_blocks"]
        for i, block in enumerate(current_blocks):
            if i >= len(prev_blocks):
                # New reasoning block — add a separator if there was a previous
                # block, then stream the full text.
                if i > 0:
                    out_reasoning("\n\n")
                if block:
                    out_reasoning(block)
                prev_blocks.append(block)
            elif block.startswith(prev_blocks[i]):
                # Existing block grew — stream the delta.
                delta = block[len(prev_blocks[i]):]
                if delta:
                    out_reasoning(delta)
                prev_blocks[i] = block
            elif len(block) > len(prev_blocks[i]):
                # Non-prefix but longer (block flipped mid-stream): accept
                # without re-printing.
                prev_blocks[i] = block
        state["reasoning_blocks"] = prev_blocks

        # Surface tool activity: first appearance emits a line; the
        # executing->done transition emits the result. Also accumulate the
        # structured result for --json capture (state["tool_results"]).
        tool_done = state["tool_done"]
        tool_results = state["tool_results"]
        results_by_name = {r.get("name"): r for r in tool_results if isinstance(r, dict)}
        for ev in extract_tool_events(content):
            name = ev.get("name") or "?"
            prev = tool_done.get(name)
            if prev is None:
                # first time we see this tool
                out_tool(format_tool_line(ev))
                tool_done[name] = bool(ev.get("done"))
            elif ev.get("done") and not prev:
                out_tool(format_tool_line(ev))
                tool_done[name] = True
            # Keep the latest structured result for this tool (name + result).
            if name in results_by_name:
                results_by_name[name]["result"] = ev.get("result")
                results_by_name[name]["done"] = bool(ev.get("done"))
            else:
                entry = {"name": name, "result": ev.get("result"), "done": bool(ev.get("done"))}
                results_by_name[name] = entry
                tool_results.append(entry)

        # Stream the answer prose via prefix-diff. The server emits the running
        # answer as content grows; the final ``done`` event re-sends the complete
        # content. We only print NEW text.
        #
        # When a tool block flips (Executing -> done) the serialized prose can
        # shift so the new answer is NOT a prefix-extension of the previous - in
        # that case we DON'T print anything (the user already has the streamed
        # fragments) and, crucially, we DON'T update state["answer"]: keeping
        # the last prefix-consistent value keeps later deltas comparable. The
        # final ``done`` event's content becomes the return value via the delta
        # path below (it extends fine, with an empty delta).
        answer = extract_answer_text(content)
        if answer.startswith(state["answer"]):
            delta = answer[len(state["answer"]):]
            if delta:
                out_text(delta)
            state["answer"] = answer
        elif len(answer) > len(state["answer"]):
            # Non-prefix but longer final answer: accept it as the authoritative
            # return value without double-printing streamed text.
            state["answer"] = answer

        if data.get("done"):
            # Capture the full serialized content on the terminal event so
            # callers (--save) can persist it with <details> blocks intact —
            # the web UI renders reasoning/tool_calls FROM those blocks.
            state["raw_content"] = content
            done.set()

    @sio.on("chat-events")
    async def _on_chat_events(payload, cb=None):  # type: ignore[no-untyped-def]
        # socketio may call this from its own event loop thread; the per-event
        # work is sync, so just run it. cb is an optional ack callback.
        try:
            _handle_event(payload)
        except Exception as exc:  # noqa: BLE001 - deliberate: keep the socket loop alive on any tool-event error
            state["error"] = f"error handling chat event: {exc}"
            done.set()
        if cb:
            with contextlib.suppress(Exception):
                await cb(True)

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
            # Some builds still stream over HTTP for this path; surface a clear msg.
            raise APIError(
                f"unexpected chat ack: {ack!r}; server did not start a background task"
            )

        # ---- wait for completion ----
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except _AsyncTimeoutError:
            state["error"] = f"timed out after {timeout}s"
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
    }
