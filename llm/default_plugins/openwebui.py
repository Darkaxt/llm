"""Open WebUI provider for LLM.

This built-in plugin keeps Open WebUI-specific auth and transport isolated from
LLM's conversation engine. It uses openwebui-sdk for email/password sign-in,
model discovery, streaming, reasoning and the Socket.IO tool execution path.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import click
import httpx2
import llm
from llm.default_plugins.openwebui_socket import run_chat_with_tools_with_files
from llm.parts import AttachmentPart, ReasoningPart, StreamEvent, TextPart, ToolResultPart
from openwebui_sdk import ChatResult, OpenWebUIClient
from openwebui_sdk.errors import APIError, AuthError

CONFIG_FILENAME = "openwebui.json"


def _config_path() -> Path:
    return llm.user_dir() / CONFIG_FILENAME


def _load_config() -> dict[str, Any] | None:
    path = _config_path()
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _save_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def _client(config: dict[str, Any]) -> OpenWebUIClient:
    url = config.get("url")
    token = config.get("token")
    if not url or not token:
        raise llm.ModelError(
            "Open WebUI is not configured. Run: llm openwebui login --url URL --email EMAIL"
        )
    return OpenWebUIClient(base_url=str(url), token=str(token))


def _model_cache(client: OpenWebUIClient) -> list[dict[str, str]]:
    return [
        {"id": model.id, "name": model.name or model.id}
        for model in client.list_models()
        if model.id
    ]


def _persist_session(config: dict[str, Any], client: OpenWebUIClient) -> dict[str, Any]:
    session = client.session()
    config = dict(config)
    config["token"] = session.token
    config["user"] = {
        "id": session.user_id,
        "email": session.email,
        "name": session.name,
        "role": session.role,
    }
    _save_config(config)
    return config


def _message_content(message: Any) -> str:
    chunks: list[str] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            chunks.append(part.text)
        elif isinstance(part, ToolResultPart):
            chunks.append(part.output)
        elif isinstance(part, ReasoningPart):
            # Provider-side reasoning is not part of the visible conversation
            # context and should not be sent back as ordinary text.
            continue
        elif isinstance(part, AttachmentPart):
            # Attachments are uploaded separately and referenced through the
            # Open WebUI request's top-level "files" field.
            continue
    return "\n".join(chunk for chunk in chunks if chunk)


def _attachment_filename(attachment: llm.Attachment, index: int) -> str:
    if attachment.path:
        return Path(attachment.path).name
    if attachment.url:
        name = Path(urlparse(attachment.url).path).name
        if name:
            return name
    suffix = {
        "text/markdown": ".md",
        "application/json": ".json",
        "application/pdf": ".pdf",
    }.get(attachment.type or "", "")
    return f"attachment-{index}{suffix}"


def _file_processing_timeout() -> float:
    raw = os.environ.get("LLM_OPENWEBUI_FILE_TIMEOUT", "600")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise llm.ModelError(
            "LLM_OPENWEBUI_FILE_TIMEOUT must be a number of seconds"
        ) from exc
    if timeout <= 0:
        raise llm.ModelError("LLM_OPENWEBUI_FILE_TIMEOUT must be greater than zero")
    return timeout


def _wait_for_file_processing(
    http: httpx2.Client,
    client: OpenWebUIClient,
    file_id: str,
    filename: str,
) -> None:
    """Wait for Open WebUI's background extraction/RAG processing.

    The browser uses /process/status after the upload POST instead of keeping
    the upload request open. Polling the non-streaming status endpoint gives us
    the same semantics without tying a long-running SSE connection to the CLI.
    """
    deadline = time.monotonic() + _file_processing_timeout()
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {client.token}",
    }

    last_status = None
    while True:
        if time.monotonic() >= deadline:
            raise llm.ModelError(
                f"Open WebUI timed out processing {filename} "
                f"after {_file_processing_timeout():g}s"
            )

        try:
            status_response = http.get(
                f"{client.base_url}/api/v1/files/{file_id}/process/status",
                params={"stream": "false"},
                headers=headers,
            )
            status_response.raise_for_status()
            payload = status_response.json()
        except Exception as exc:
            raise llm.ModelError(
                f"Open WebUI failed while checking processing status for "
                f"{filename}: {exc}"
            ) from exc

        status = payload.get("status") if isinstance(payload, dict) else None
        if status != last_status:
            click.echo(
                f"[Open WebUI] {filename}: {status or 'waiting'}",
                err=True,
            )
            last_status = status
        if status == "completed":
            return
        if status == "failed":
            detail = None
            try:
                file_response = http.get(
                    f"{client.base_url}/api/v1/files/{file_id}",
                    headers=headers,
                )
                file_response.raise_for_status()
                file_payload = file_response.json()
                if isinstance(file_payload, dict):
                    data = file_payload.get("data")
                    if isinstance(data, dict):
                        detail = data.get("error")
            except Exception:
                pass
            suffix = f": {detail}" if detail else ""
            raise llm.ModelError(
                f"Open WebUI failed to process {filename}{suffix}"
            )
        if status not in ("pending", "processing", None):
            raise llm.ModelError(
                f"Open WebUI returned unexpected processing status "
                f"{status!r} for {filename}"
            )

        time.sleep(1)


def _upload_attachment(
    client: OpenWebUIClient,
    attachment: llm.Attachment,
    index: int,
) -> dict[str, Any]:
    """Upload one LLM attachment using Open WebUI's native file API.

    Documents follow the browser flow: the upload POST returns quickly while
    extraction/RAG processing continues in the background, then we wait on the
    file process-status endpoint before starting model inference. Raster images
    are stored without document processing.
    """
    content_type = attachment.type
    if not content_type:
        try:
            content_type = attachment.resolve_type()
        except Exception as exc:
            raise llm.ModelError(f"Could not determine attachment type: {exc}") from exc
    content_type = content_type or "application/octet-stream"
    filename = _attachment_filename(attachment, index)
    try:
        content = attachment.content_bytes()
    except Exception as exc:
        raise llm.ModelError(f"Could not read attachment {filename}: {exc}") from exc
    if not content:
        raise llm.ModelError(f"Attachment {filename} is empty")

    is_image = content_type.startswith("image/")
    params = {
        "process": "false" if is_image else "true",
        # Match the Open WebUI browser: return the uploaded file record first,
        # then observe processing separately through /process/status.
        "process_in_background": "true",
    }
    upload_timeout = max(float(client.timeout), 120.0)
    click.echo(f"[Open WebUI] {filename}: uploading", err=True)
    try:
        with httpx2.Client(trust_env=True, timeout=upload_timeout) as http:
            result = http.post(
                f"{client.base_url}/api/v1/files/",
                params=params,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {client.token}",
                },
                files={"file": (filename, content, content_type)},
            )
            result.raise_for_status()
            uploaded = result.json()

            if not isinstance(uploaded, dict) or not uploaded.get("id"):
                raise llm.ModelError(
                    f"Open WebUI returned an invalid upload response for {filename}"
                )

            if not is_image:
                _wait_for_file_processing(
                    http,
                    client,
                    str(uploaded["id"]),
                    filename,
                )
    except llm.ModelError:
        raise
    except Exception as exc:
        raise llm.ModelError(f"Open WebUI failed to upload {filename}: {exc}") from exc

    meta = uploaded.get("meta") if isinstance(uploaded.get("meta"), dict) else {}
    effective_type = meta.get("content_type") or content_type
    item: dict[str, Any] = {
        "type": "image" if str(effective_type).startswith("image/") else "file",
        "file": uploaded,
        "id": str(uploaded["id"]),
        "url": str(uploaded["id"]),
        "name": uploaded.get("filename") or filename,
        "status": "uploaded",
        "content_type": effective_type,
        "size": meta.get("size", len(content)),
        "collection_name": meta.get("collection_name")
        or uploaded.get("collection_name")
        or "",
    }
    # LLM's -a semantics mean "send this attachment", not "search a few chunks".
    # Full context best matches that expectation for document attachments.
    if item["type"] == "file":
        item["context"] = "full"
    return item


def _prepare_openwebui_request(
    prompt: llm.Prompt,
    client: OpenWebUIClient,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = []
    files_by_id: dict[str, dict[str, Any]] = {}

    for message in prompt.messages:
        has_attachment = False
        for index, part in enumerate(message.parts):
            if not isinstance(part, AttachmentPart) or part.attachment is None:
                continue
            has_attachment = True
            metadata = part.provider_metadata or {}
            cached = metadata.get("openwebui_file")
            cached_url = metadata.get("openwebui_url")
            if isinstance(cached, dict) and cached.get("id") and cached_url == client.base_url:
                item = cached
            else:
                item = _upload_attachment(client, part.attachment, index)
                part.provider_metadata = {
                    **metadata,
                    "openwebui_file": item,
                    "openwebui_url": client.base_url,
                }
            files_by_id[str(item["id"])] = item

        content = _message_content(message)
        if content or has_attachment:
            messages.append({"role": message.role, "content": content})

    return messages, list(files_by_id.values())


class OpenWebUIModel(llm.Model):
    """One model exposed by the configured Open WebUI instance."""

    class Options(llm.Options):
        temperature: float | None = None
        openwebui_tools: bool = True

    def __init__(self, remote_model_id: str, display_name: str | None = None):
        self.remote_model_id = remote_model_id
        self.display_name = display_name or remote_model_id
        self.model_id = f"openwebui/{remote_model_id}"

    # Open WebUI's /api/v1/files endpoint accepts general file uploads and
    # applies the deployment's own allowed-extension / size policy. LLM's base
    # Model class otherwise rejects every attachment unless it is enumerated in
    # attachment_types, which is too restrictive for a dynamic Open WebUI
    # backend (Markdown, JSON, ZIP bundles, PDFs, source files, etc.).
    def _validate_attachments(
        self, attachments: list[llm.Attachment] | None = None
    ) -> None:
        for attachment in attachments or []:
            # Resolve early so path/stdin attachments still fail with a useful
            # error if LLM cannot determine a MIME type. The Open WebUI server
            # remains authoritative for whether that type/extension is allowed.
            try:
                attachment.resolve_type()
            except Exception as exc:
                raise ValueError(f"Could not determine attachment type: {exc}") from exc

    def __str__(self) -> str:
        return f"Open WebUI: {self.display_name}"

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None,
    ) -> Iterator[str | StreamEvent]:
        config = _load_config()
        if not config:
            raise llm.ModelError(
                "Open WebUI is not configured. Run: llm openwebui login --url URL --email EMAIL"
            )

        client = _client(config)
        messages, attached_files = _prepare_openwebui_request(prompt, client)

        try:
            tool_ids = client.resolve_tools(
                self.remote_model_id,
                no_tools=not prompt.options.openwebui_tools,
            )
        except (APIError, AuthError) as exc:
            raise llm.ModelError(str(exc)) from exc

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        tool_activity: list[str] = []
        status_activity: list[str] = []

        def on_text(fragment: str) -> None:
            events.put(("text", fragment))

        def on_reasoning(fragment: str) -> None:
            events.put(("reasoning", fragment))

        def on_tool(line: str) -> None:
            tool_activity.append(line)

        def on_status(line: str) -> None:
            status_activity.append(line)

        def worker() -> None:
            try:
                if tool_ids and attached_files:
                    # openwebui-sdk 0.1.1 does not yet expose its top-level
                    # "files" field on the Socket.IO tool path. Use the vendored
                    # compatibility runner so attachments and model-attached
                    # tools work together instead of silently dropping either.
                    data = __import__("asyncio").run(
                        run_chat_with_tools_with_files(
                            base_url=client.base_url,
                            token=client.token or "",
                            model=self.remote_model_id,
                            messages=messages,
                            tool_ids=tool_ids,
                            files=attached_files,
                            on_text=on_text,
                            on_reasoning=on_reasoning,
                            on_tool=on_tool,
                            on_status=on_status,
                        )
                    )
                    result = ChatResult(
                        answer=data.get("answer", ""),
                        reasoning=data.get("reasoning"),
                        tool_calls=data.get("tool_calls", []),
                        raw_content=data.get("raw_content", ""),
                    )
                else:
                    result = client.run_chat(
                        model=self.remote_model_id,
                        messages=messages,
                        tool_ids=tool_ids,
                        temperature=prompt.options.temperature,
                        extra={"files": attached_files} if attached_files else None,
                        on_text=on_text,
                        on_reasoning=on_reasoning,
                        on_tool=on_tool,
                        on_status=on_status,
                    )
                events.put(("done", result))
            except Exception as exc:  # handed back to the main iterator
                events.put(("error", exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            kind, payload = events.get()
            if kind == "text":
                yield str(payload)
            elif kind == "reasoning":
                if not prompt.hide_reasoning:
                    yield StreamEvent(type="reasoning", chunk=str(payload))
            elif kind == "error":
                if isinstance(payload, (APIError, AuthError)):
                    raise llm.ModelError(str(payload)) from payload
                raise payload
            elif kind == "done":
                result = payload
                response.response_json = {
                    "provider": "openwebui",
                    "remote_model": self.remote_model_id,
                    "reasoning": result.reasoning,
                    "tool_calls": result.tool_calls,
                    "tool_activity": tool_activity,
                    "status_activity": status_activity,
                }
                break


@llm.hookimpl
def register_models(register):
    config = _load_config()
    if not config:
        return
    for item in config.get("models", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        remote_id = str(item["id"])
        model = OpenWebUIModel(remote_id, item.get("name"))
        register(model, aliases=[f"owui/{remote_id}"])


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="openwebui")
    def openwebui_group():
        """Configure and inspect the Open WebUI provider."""

    @openwebui_group.command(name="login")
    @click.option("--url", required=True, help="Open WebUI base URL")
    @click.option("--email", required=True, help="Open WebUI email address")
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=False)
    def login(url: str, email: str, password: str):
        """Sign in with email/password and cache the visible model catalogue."""
        client = OpenWebUIClient(base_url=url)
        try:
            session = client.signin(email, password)
            models = _model_cache(client)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc

        config = {
            "url": url.rstrip("/"),
            "email": email,
            "token": session.token,
            "user": {
                "id": session.user_id,
                "email": session.email,
                "name": session.name,
                "role": session.role,
            },
            "models": models,
        }
        _save_config(config)
        click.echo(
            f"Saved Open WebUI session for {session.email or email}; "
            f"{len(models)} model(s) cached."
        )

    @openwebui_group.command(name="whoami")
    def whoami():
        """Validate the current session and show its identity."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            config = _persist_session(config, client)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(config.get("user", {}), indent=2, ensure_ascii=False))

    @openwebui_group.command(name="sync")
    def sync():
        """Refresh the session and cached model catalogue."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            config = _persist_session(config, client)
            config["models"] = _model_cache(client)
            _save_config(config)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Cached {len(config['models'])} Open WebUI model(s).")

    @openwebui_group.command(name="models")
    def models():
        """Show the cached Open WebUI model catalogue."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        for item in config.get("models", []):
            click.echo(
                f"openwebui/{item.get('id')}\t{item.get('name') or item.get('id')}"
            )
