"""Open WebUI provider for LLM.

This built-in plugin keeps Open WebUI-specific auth and transport isolated from
LLM's conversation engine. It uses openwebui-sdk for email/password sign-in,
model discovery, streaming, reasoning and the Socket.IO tool execution path.
"""

from __future__ import annotations

import io
import json
import os
import queue
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Literal
from urllib.parse import urlparse

import click
import httpx2
import llm
from llm.default_plugins.openwebui_socket import run_chat_with_tools_with_files
from llm.parts import AttachmentPart, ReasoningPart, StreamEvent, TextPart, ToolResultPart
from openwebui_sdk import ChatResult, OpenWebUIClient
from openwebui_sdk.errors import APIError, AuthError

CONFIG_FILENAME = "openwebui.json"


class _OpenWebUIStatusBar:
    """One transient terminal line for Open WebUI transport/meta activity."""

    def __init__(self) -> None:
        self.enabled = bool(
            getattr(sys.stderr, "isatty", lambda: False)()
            and getattr(sys.stdout, "isatty", lambda: False)()
        )
        self.visible = False
        self.current = ""

    def _width(self) -> int:
        return max(20, shutil.get_terminal_size((120, 24)).columns)

    def _render_text(self, text: str, *, kind: str = "status") -> str:
        normalized = " ".join(str(text).split())
        if kind == "tool":
            normalized = f"tool · {normalized}"
        prefix = "Open WebUI │ "
        width = self._width()
        available = max(8, width - len(prefix) - 1)
        if len(normalized) > available:
            normalized = normalized[: max(1, available - 1)] + "…"
        return prefix + normalized

    def update(self, text: str, *, kind: str = "status") -> None:
        self.current = str(text)
        if not self.enabled:
            prefix = "[Open WebUI tool]" if kind == "tool" else "[Open WebUI]"
            click.echo(f"{prefix} {text}", err=True)
            return

        width = self._width()
        line = self._render_text(text, kind=kind)
        # Use only carriage returns/spaces rather than ANSI cursor control so
        # this remains reliable in Windows Terminal/PowerShell.
        sys.stderr.write("\r" + (" " * (width - 1)) + "\r" + line)
        sys.stderr.flush()
        self.visible = True

    def clear(self) -> None:
        if not self.enabled or not self.visible:
            return
        width = self._width()
        sys.stderr.write("\r" + (" " * (width - 1)) + "\r")
        sys.stderr.flush()
        self.visible = False


def _escape_pressed() -> bool:
    """Return True when Escape was pressed during generation on Windows.

    GetAsyncKeyState lets us detect Escape without consuming type-ahead from
    stdin. Ctrl+C remains handled by Python's normal KeyboardInterrupt path.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x0001)
    except Exception:
        return False


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


def _chat_timeout() -> int:
    raw = os.environ.get("LLM_OPENWEBUI_CHAT_TIMEOUT", "600")
    try:
        timeout = int(float(raw))
    except ValueError as exc:
        raise llm.ModelError(
            "LLM_OPENWEBUI_CHAT_TIMEOUT must be a number of seconds"
        ) from exc
    if timeout <= 0:
        raise llm.ModelError("LLM_OPENWEBUI_CHAT_TIMEOUT must be greater than zero")
    return timeout


def _client(config: dict[str, Any]) -> OpenWebUIClient:
    url = config.get("url")
    token = config.get("token")
    if not url or not token:
        raise llm.ModelError(
            "Open WebUI is not configured. Run: llm openwebui login --url URL --email EMAIL"
        )
    return OpenWebUIClient(
        base_url=str(url),
        token=str(token),
        timeout=_chat_timeout(),
    )


def _model_cache(client: OpenWebUIClient) -> list[dict[str, str]]:
    return [
        {"id": model.id, "name": model.name or model.id}
        for model in client.list_models()
        if model.id
    ]


def _enabled_tool_ids(config: dict[str, Any]) -> list[str]:
    raw = config.get("enabled_tool_ids", [])
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if value]


def _tool_kind(tool_id: str) -> str:
    if tool_id.startswith("server:mcp:"):
        return "mcp"
    if tool_id.startswith("server:"):
        return "server"
    return "tool"


def _resolve_tool_selector(client: OpenWebUIClient, selector: str):
    tools = client.list_tools()
    if not tools:
        raise click.ClickException("Open WebUI returned no available tools")

    exact_id = [tool for tool in tools if tool.id == selector]
    if exact_id:
        return exact_id[0]

    folded = selector.casefold()
    exact_name = [tool for tool in tools if tool.name.casefold() == folded]
    if len(exact_name) == 1:
        return exact_name[0]
    if len(exact_name) > 1:
        matches = ", ".join(f"{tool.name} ({tool.id})" for tool in exact_name)
        raise click.ClickException(f"Tool name is ambiguous: {matches}")

    partial = [
        tool
        for tool in tools
        if folded in tool.id.casefold() or folded in tool.name.casefold()
    ]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        matches = ", ".join(f"{tool.name} ({tool.id})" for tool in partial)
        raise click.ClickException(
            f"Tool selector {selector!r} is ambiguous: {matches}"
        )
    raise click.ClickException(f"No Open WebUI tool matches {selector!r}")


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


_ARCHIVE_SIDECAR_TYPES = {
    "rule-index.json": "application/json",
    "splunk-rules.jsonl": "application/x-ndjson",
    "macros.json": "application/json",
}
_ARCHIVE_SIDECAR_MAX_BYTES = 64 * 1024 * 1024


def _normalized_archive_path(name: str) -> PurePosixPath:
    # Archive formats use POSIX separators by convention, but normalize
    # backslashes too so archives produced on Windows behave consistently.
    return PurePosixPath(str(name).replace("\\", "/").lstrip("/"))


def _select_sidecar_members(member_names: list[str]) -> dict[str, str]:
    """Resolve expected sidecars without silently collapsing path collisions.

    Prefer exactly one directory that contains a complete sidecar set. If there
    is no complete set, globally unique sidecars are allowed. Any remaining
    duplicate basename is ambiguous and therefore rejected.
    """
    by_parent: dict[str, dict[str, list[str]]] = {}
    by_basename: dict[str, list[str]] = {}

    for raw_name in member_names:
        path = _normalized_archive_path(raw_name)
        basename = path.name.lower()
        if basename not in _ARCHIVE_SIDECAR_TYPES:
            continue
        parent = path.parent.as_posix()
        by_parent.setdefault(parent, {}).setdefault(basename, []).append(raw_name)
        by_basename.setdefault(basename, []).append(raw_name)

    complete_roots: list[str] = []
    for parent, members in by_parent.items():
        if all(
            len(members.get(expected_name, [])) == 1
            for expected_name in _ARCHIVE_SIDECAR_TYPES
        ):
            complete_roots.append(parent)

    if len(complete_roots) > 1:
        roots = ", ".join(repr(root or ".") for root in sorted(complete_roots))
        raise llm.ModelError(
            "Archive contains multiple complete TIDE sidecar sets under "
            f"different paths: {roots}"
        )

    if len(complete_roots) == 1:
        root = complete_roots[0]
        members = by_parent[root]
        return {
            expected_name: members[expected_name][0]
            for expected_name in _ARCHIVE_SIDECAR_TYPES
        }

    ambiguous = {
        expected_name: candidates
        for expected_name, candidates in by_basename.items()
        if len(candidates) > 1
    }
    if ambiguous:
        detail = "; ".join(
            f"{name}: {', '.join(paths)}"
            for name, paths in sorted(ambiguous.items())
        )
        raise llm.ModelError(
            "Archive contains ambiguous TIDE sidecar filenames under "
            f"different paths: {detail}"
        )

    # Partial but unambiguous sets are still useful and preserve the previous
    # behavior for older/minimal bundles.
    return {
        expected_name: candidates[0]
        for expected_name in _ARCHIVE_SIDECAR_TYPES
        if (candidates := by_basename.get(expected_name))
    }


def _zip_sidecar_attachments(
    attachment: llm.Attachment,
    *,
    filename: str | None = None,
) -> list[tuple[str, llm.Attachment]]:
    filename = filename or _attachment_filename(attachment, 0)
    content_type = attachment.type or ""
    if not (
        filename.lower().endswith(".zip")
        or content_type in {"application/zip", "application/x-zip-compressed"}
    ):
        return []

    try:
        content = attachment.content_bytes()
    except Exception as exc:
        raise llm.ModelError(
            f"Could not read ZIP attachment {filename}: {exc}"
        ) from exc
    if not content:
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            selected = _select_sidecar_members([info.filename for info in infos])
            if not selected:
                return []

            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in infos:
                infos_by_name.setdefault(info.filename, []).append(info)

            sidecars: list[tuple[str, llm.Attachment]] = []
            for expected_name, mime in _ARCHIVE_SIDECAR_TYPES.items():
                member_name = selected.get(expected_name)
                if member_name is None:
                    continue
                matching_infos = infos_by_name.get(member_name, [])
                if len(matching_infos) != 1:
                    raise llm.ModelError(
                        f"Archive member {member_name!r} is duplicated and ambiguous"
                    )
                info = matching_infos[0]
                if info.file_size > _ARCHIVE_SIDECAR_MAX_BYTES:
                    raise llm.ModelError(
                        f"Archive sidecar {member_name} is too large "
                        f"({info.file_size} bytes)"
                    )
                sidecars.append(
                    (
                        expected_name,
                        llm.Attachment(
                            type=mime,
                            content=archive.read(info),
                        ),
                    )
                )
            return sidecars
    except zipfile.BadZipFile:
        return []


def _sevenzip_sidecar_attachments(
    attachment: llm.Attachment,
    *,
    filename: str | None = None,
) -> list[tuple[str, llm.Attachment]]:
    filename = filename or _attachment_filename(attachment, 0)
    content_type = attachment.type or ""
    if not (
        filename.lower().endswith(".7z")
        or content_type in {
            "application/x-7z-compressed",
            "application/7z",
        }
    ):
        return []

    try:
        import py7zr
    except ImportError as exc:
        raise llm.ModelError(
            "7z attachment support requires py7zr; reinstall the project "
            "with: python -m pip install -e ."
        ) from exc

    try:
        content = attachment.content_bytes()
    except Exception as exc:
        raise llm.ModelError(
            f"Could not read 7z attachment {filename}: {exc}"
        ) from exc
    if not content:
        return []

    class _MemoryIO(py7zr.Py7zIO):
        def __init__(self) -> None:
            self.buffer = io.BytesIO()

        def write(self, data):
            return self.buffer.write(data)

        def read(self, size=None):
            return self.buffer.read(-1 if size is None else size)

        def seek(self, offset, whence=0):
            return self.buffer.seek(offset, whence)

        def flush(self) -> None:
            return None

        def size(self) -> int:
            return self.buffer.getbuffer().nbytes

        def getvalue(self) -> bytes:
            return self.buffer.getvalue()

    class _MemoryFactory(py7zr.WriterFactory):
        def __init__(self) -> None:
            self.products: dict[str, _MemoryIO] = {}

        def create(self, filename):
            product = _MemoryIO()
            self.products[str(filename).replace("\\", "/")] = product
            return product

    try:
        archive_buffer = io.BytesIO(content)
        with py7zr.SevenZipFile(archive_buffer, mode="r") as archive:
            if archive.needs_password():
                raise llm.ModelError(
                    f"7z attachment {filename} is password-protected"
                )

            infos = [info for info in archive.list() if not info.is_directory]
            selected = _select_sidecar_members([info.filename for info in infos])
            if not selected:
                return []

            info_by_name = {info.filename: info for info in infos}
            for member_name in selected.values():
                info = info_by_name.get(member_name)
                if info is None:
                    raise llm.ModelError(
                        f"7z member {member_name!r} disappeared while reading archive"
                    )
                if int(info.uncompressed or 0) > _ARCHIVE_SIDECAR_MAX_BYTES:
                    raise llm.ModelError(
                        f"Archive sidecar {member_name} is too large "
                        f"({info.uncompressed} bytes)"
                    )

            targets = list(selected.values())
            # py7zr may require explicit parent directory entries when extracting
            # nested files. Include only parent entries that actually exist.
            member_name_set = {info.filename for info in archive.list()}
            for member_name in list(targets):
                path = _normalized_archive_path(member_name)
                for parent in path.parents:
                    parent_name = parent.as_posix()
                    if parent_name in ("", "."):
                        continue
                    if parent_name in member_name_set and parent_name not in targets:
                        targets.append(parent_name)

            factory = _MemoryFactory()
            archive.extract(targets=targets, factory=factory)

        sidecars: list[tuple[str, llm.Attachment]] = []
        for expected_name, mime in _ARCHIVE_SIDECAR_TYPES.items():
            member_name = selected.get(expected_name)
            if member_name is None:
                continue
            normalized_member = _normalized_archive_path(member_name).as_posix()
            matches = [
                product
                for product_name, product in factory.products.items()
                if _normalized_archive_path(product_name).as_posix()
                == normalized_member
                or _normalized_archive_path(product_name).as_posix().endswith(
                    "/" + normalized_member
                )
            ]
            if len(matches) != 1:
                raise llm.ModelError(
                    f"Could not uniquely extract 7z sidecar {member_name!r}"
                )
            sidecars.append(
                (
                    expected_name,
                    llm.Attachment(type=mime, content=matches[0].getvalue()),
                )
            )
        return sidecars
    except llm.ModelError:
        raise
    except py7zr.Bad7zFile:
        return []
    except Exception as exc:
        raise llm.ModelError(
            f"Failed to read 7z attachment {filename}: {exc}"
        ) from exc


def _archive_sidecar_attachments(
    attachment: llm.Attachment,
    *,
    filename: str | None = None,
) -> list[tuple[str, llm.Attachment]]:
    filename = filename or _attachment_filename(attachment, 0)
    lower_name = filename.lower()
    content_type = attachment.type or ""

    if lower_name.endswith(".7z") or content_type in {
        "application/x-7z-compressed",
        "application/7z",
    }:
        return _sevenzip_sidecar_attachments(
            attachment,
            filename=filename,
        )

    return _zip_sidecar_attachments(
        attachment,
        filename=filename,
    )


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


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _activity_bar(tick: int, width: int = 18) -> str:
    """Return an indeterminate moving marker, not a fake percentage."""
    if width < 3:
        width = 3
    cycle = (width - 1) * 2
    pos = tick % cycle
    if pos >= width:
        pos = cycle - pos
    chars = [" "] * width
    chars[pos] = "="
    return "[" + "".join(chars) + "]"


def _render_file_progress(
    *,
    filename: str,
    status: str,
    started: float,
    tick: int,
    ordinal: int,
    total: int,
    done: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> None:
    elapsed = _format_elapsed(time.monotonic() - started)
    if on_status is not None:
        on_status(
            f"attachments {ordinal}/{total} · {filename} · {status} · {elapsed}"
        )
        return

    bar = "[" + "=" * 18 + "]" if done else _activity_bar(tick)
    line = (
        f"[Open WebUI] {ordinal}/{total} {bar} "
        f"{filename}: {status} · {elapsed}"
    )
    if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        click.echo("\r" + line.ljust(120), nl=done, err=True)
    else:
        click.echo(line, err=True)


def _wait_for_file_processing(
    http: httpx2.Client,
    client: OpenWebUIClient,
    file_id: str,
    filename: str,
    *,
    ordinal: int = 1,
    total: int = 1,
    started: float | None = None,
    on_status: Callable[[str], None] | None = None,
    interactive_status: bool = False,
) -> None:
    """Wait for Open WebUI's background extraction/RAG processing.

    The browser uses /process/status after the upload POST instead of keeping
    the upload request open. Polling the non-streaming status endpoint gives us
    the same semantics without tying a long-running SSE connection to the CLI.
    """
    started = started if started is not None else time.monotonic()
    deadline = started + _file_processing_timeout()
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {client.token}",
    }

    last_status = None
    tick = 0
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
        display_status = status or "waiting"
        terminal_updates = interactive_status or (
            on_status is None
            and hasattr(sys.stderr, "isatty")
            and sys.stderr.isatty()
        )
        if status != last_status or terminal_updates:
            _render_file_progress(
                filename=filename,
                status=display_status,
                started=started,
                tick=tick,
                ordinal=ordinal,
                total=total,
                done=status == "completed",
                on_status=on_status,
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

        tick += 1
        time.sleep(1)


def _upload_attachment(
    client: OpenWebUIClient,
    attachment: llm.Attachment,
    index: int,
    *,
    ordinal: int = 1,
    total: int = 1,
    on_status: Callable[[str], None] | None = None,
    interactive_status: bool = False,
    filename_override: str | None = None,
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
    filename = filename_override or _attachment_filename(attachment, index)
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
    started = time.monotonic()
    _render_file_progress(
        filename=filename,
        status="uploading",
        started=started,
        tick=0,
        ordinal=ordinal,
        total=total,
        on_status=on_status,
    )
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
                    ordinal=ordinal,
                    total=total,
                    started=started,
                    on_status=on_status,
                    interactive_status=interactive_status,
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
    return item


_TEXT_FULL_CONTEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".csv",
    ".tsv",
    ".py",
    ".ps1",
    ".sh",
    ".spl",
    ".conf",
    ".toml",
    ".ini",
}
_FULL_CONTEXT_PER_FILE_BYTES = 128 * 1024
_FULL_CONTEXT_TOTAL_BYTES = 256 * 1024


def _is_textual_attachment(item: dict[str, Any]) -> bool:
    content_type = str(item.get("content_type") or "").lower()
    if content_type.startswith("text/"):
        return True
    if content_type in {
        "application/json",
        "application/x-ndjson",
        "application/ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
    }:
        return True
    return Path(str(item.get("name") or "")).suffix.lower() in _TEXT_FULL_CONTEXT_EXTENSIONS


def _apply_attachment_context_policy(
    files: list[dict[str, Any]],
    mode: Literal["auto", "full", "rag"],
    *,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Choose full-context vs Open WebUI retrieval for uploaded files.

    Full mode mirrors the browser manual Full Context toggle and can consume
    the model window very quickly. RAG mode always uses chunked retrieval.
    Auto mode uses full context only for small textual files, with both a
    per-file and aggregate byte budget; archives/binary/large files use RAG.
    """
    total_full_bytes = 0
    decisions: list[str] = []

    for item in files:
        if item.get("type") != "file":
            continue

        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0

        use_full = False
        if mode == "full":
            use_full = True
        elif mode == "auto":
            use_full = (
                _is_textual_attachment(item)
                and size > 0
                and size <= _FULL_CONTEXT_PER_FILE_BYTES
                and total_full_bytes + size <= _FULL_CONTEXT_TOTAL_BYTES
            )

        if use_full:
            item["context"] = "full"
            total_full_bytes += size
            decision = "full"
        else:
            item.pop("context", None)
            decision = "rag"

        decisions.append(f"{item.get('name') or item.get('id')}={decision}")

    if decisions:
        message = "attachment context · " + " · ".join(decisions)
        if on_status is not None:
            on_status(message)
        else:
            click.echo("[Open WebUI] " + message, err=True)


def _prepare_openwebui_request(
    prompt: llm.Prompt,
    client: OpenWebUIClient,
    *,
    on_status: Callable[[str], None] | None = None,
    interactive_status: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = []
    files_by_id: dict[str, dict[str, Any]] = {}

    # Build the upload plan first so progress counts reflect archive expansion.
    # Cached sidecars are reused without re-reading/re-uploading the ZIP.
    sidecar_plans: dict[int, list[tuple[str, llm.Attachment]]] = {}
    total_attachments = 0
    for message in prompt.messages:
        for part in message.parts:
            if not isinstance(part, AttachmentPart) or part.attachment is None:
                continue
            metadata = part.provider_metadata or {}
            cached_sidecars = metadata.get("openwebui_sidecars")
            cached_url = metadata.get("openwebui_url")
            if (
                isinstance(cached_sidecars, list)
                and cached_sidecars
                and cached_url == client.base_url
                and all(isinstance(item, dict) and item.get("id") for item in cached_sidecars)
            ):
                total_attachments += len(cached_sidecars)
                continue

            filename = _attachment_filename(part.attachment, 0)
            sidecars = _archive_sidecar_attachments(
                part.attachment,
                filename=filename,
            )
            if sidecars:
                sidecar_plans[id(part)] = sidecars
                total_attachments += len(sidecars)
            else:
                total_attachments += 1

    attachment_ordinal = 0

    for message in prompt.messages:
        has_attachment = False
        for index, part in enumerate(message.parts):
            if not isinstance(part, AttachmentPart) or part.attachment is None:
                continue
            has_attachment = True
            metadata = part.provider_metadata or {}
            cached_url = metadata.get("openwebui_url")
            archive_filename = _attachment_filename(part.attachment, index)

            cached_sidecars = metadata.get("openwebui_sidecars")
            if (
                isinstance(cached_sidecars, list)
                and cached_sidecars
                and cached_url == client.base_url
                and all(isinstance(item, dict) and item.get("id") for item in cached_sidecars)
            ):
                if on_status is not None:
                    names = ", ".join(
                        str(item.get("name") or item.get("id"))
                        for item in cached_sidecars
                    )
                    on_status(
                        f"archive {archive_filename} · reusing sidecars · {names}"
                    )
                for item in cached_sidecars:
                    files_by_id[str(item["id"])] = item
                continue

            sidecars = sidecar_plans.get(id(part), [])
            if sidecars:
                if on_status is not None:
                    on_status(
                        f"archive {archive_filename} · exposing "
                        + ", ".join(name for name, _ in sidecars)
                    )

                uploaded_sidecars: list[dict[str, Any]] = []
                for sidecar_name, sidecar_attachment in sidecars:
                    attachment_ordinal += 1
                    item = _upload_attachment(
                        client,
                        sidecar_attachment,
                        index,
                        ordinal=attachment_ordinal,
                        total=total_attachments,
                        on_status=on_status,
                        interactive_status=interactive_status,
                        filename_override=sidecar_name,
                    )
                    uploaded_sidecars.append(item)
                    files_by_id[str(item["id"])] = item

                part.provider_metadata = {
                    **metadata,
                    "openwebui_sidecars": uploaded_sidecars,
                    "openwebui_sidecar_names": [
                        name for name, _ in sidecars
                    ],
                    "openwebui_url": client.base_url,
                    "openwebui_archive_replaced": True,
                }
                # Do not upload the ZIP itself. The skill's machine-readable
                # sidecars are the useful interface, and avoiding the archive
                # removes the slow/opaque ZIP processing path.
                continue

            attachment_ordinal += 1
            cached = metadata.get("openwebui_file")
            if (
                isinstance(cached, dict)
                and cached.get("id")
                and cached_url == client.base_url
            ):
                item = cached
            else:
                item = _upload_attachment(
                    client,
                    part.attachment,
                    index,
                    ordinal=attachment_ordinal,
                    total=total_attachments,
                    on_status=on_status,
                    interactive_status=interactive_status,
                )
                part.provider_metadata = {
                    **metadata,
                    "openwebui_file": item,
                    "openwebui_url": client.base_url,
                }
            files_by_id[str(item["id"])] = item

        content = _message_content(message)
        if content or has_attachment:
            messages.append({"role": message.role, "content": content})

    files = list(files_by_id.values())
    _apply_attachment_context_policy(
        files,
        prompt.options.openwebui_attachment_context,
        on_status=on_status,
    )
    return messages, files


class OpenWebUIModel(llm.Model):
    """One model exposed by the configured Open WebUI instance."""

    class Options(llm.Options):
        temperature: float | None = None
        openwebui_tools: bool = True
        openwebui_attachment_context: Literal["auto", "full", "rag"] = "auto"

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
        status_bar = _OpenWebUIStatusBar()
        status_activity: list[str] = []

        def prepare_status(line: str) -> None:
            status_activity.append(line)
            status_bar.update(line)

        try:
            messages, attached_files = _prepare_openwebui_request(
                prompt,
                client,
                on_status=prepare_status,
                interactive_status=status_bar.enabled,
            )
        except Exception:
            status_bar.clear()
            raise

        try:
            tool_ids = client.resolve_tools(
                self.remote_model_id,
                extra_tool_ids=_enabled_tool_ids(config),
                no_tools=not prompt.options.openwebui_tools,
            )
        except (APIError, AuthError) as exc:
            raise llm.ModelError(str(exc)) from exc

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        tool_activity: list[str] = []
        remote_execution: dict[str, Any] = {}

        def on_text(fragment: str) -> None:
            events.put(("text", fragment))

        def on_reasoning(fragment: str) -> None:
            events.put(("reasoning", fragment))

        def on_tool(line: str) -> None:
            tool_activity.append(line)
            events.put(("tool", line))

        def on_status(line: str) -> None:
            status_activity.append(line)
            events.put(("status", line))

        def worker() -> None:
            try:
                if tool_ids:
                    # Always use our compatibility runner for tool-enabled chats.
                    # The installed SDK still generates a bare UUID chat_id for
                    # unsaved chats, which current Open WebUI treats as a missing
                    # persisted chat and rejects with 404. Our runner mirrors the
                    # browser's temporary:<socket-id> convention and also supports
                    # attachments on the same request.
                    data = __import__("asyncio").run(
                        run_chat_with_tools_with_files(
                            base_url=client.base_url,
                            token=client.token or "",
                            model=self.remote_model_id,
                            messages=messages,
                            tool_ids=tool_ids,
                            files=attached_files,
                            timeout=client.timeout,
                            on_text=on_text,
                            on_reasoning=on_reasoning,
                            on_tool=on_tool,
                            on_status=on_status,
                        )
                    )
                    remote_execution.update(
                        {
                            "remote_chat_id": data.get("remote_chat_id"),
                            "remote_task_ids": data.get("remote_task_ids", []),
                        }
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

        output_line_open = False
        try:
            while True:
                if _escape_pressed():
                    raise KeyboardInterrupt
                try:
                    kind, payload = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "text":
                    status_bar.clear()
                    chunk = str(payload)
                    output_line_open = bool(chunk) and not chunk.endswith(("\n", "\r"))
                    yield chunk
                elif kind == "reasoning":
                    if not prompt.hide_reasoning:
                        status_bar.clear()
                        chunk = str(payload)
                        output_line_open = bool(chunk) and not chunk.endswith(("\n", "\r"))
                        yield StreamEvent(type="reasoning", chunk=chunk)
                elif kind in ("tool", "status"):
                    if output_line_open:
                        click.echo("", err=True)
                        output_line_open = False
                    status_bar.update(
                        str(payload),
                        kind="tool" if kind == "tool" else "status",
                    )
                elif kind == "error":
                    status_bar.clear()
                    if isinstance(payload, (APIError, AuthError)):
                        raise llm.ModelError(str(payload)) from payload
                    raise payload
                elif kind == "done":
                    status_bar.clear()
                    result = payload
                    response.response_json = {
                        "provider": "openwebui",
                        "remote_model": self.remote_model_id,
                        "reasoning": result.reasoning,
                        "tool_calls": result.tool_calls,
                        "tool_activity": tool_activity,
                        "status_activity": status_activity,
                        "raw_content": result.raw_content,
                        **remote_execution,
                    }
                    break
        finally:
            status_bar.clear()


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


    @openwebui_group.command(name="tools")
    def tools():
        """List Open WebUI tools and MCP servers visible to this user."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            available = client.list_tools()
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc

        enabled = set(_enabled_tool_ids(config))
        if not available:
            click.echo("No Open WebUI tools are visible to this user.")
            return
        for tool in available:
            marker = "*" if tool.id in enabled else " "
            click.echo(
                f"{marker}\t{_tool_kind(tool.id)}\t{tool.name}\t{tool.id}"
            )

    @openwebui_group.group(name="tool")
    def tool_group():
        """Enable or disable Open WebUI tools for CLI chats."""

    @tool_group.command(name="enable")
    @click.argument("selector")
    def tool_enable(selector: str):
        """Persist an Open WebUI tool/MCP server as enabled for CLI chats."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            tool = _resolve_tool_selector(client, selector)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc

        enabled = _enabled_tool_ids(config)
        if tool.id not in enabled:
            enabled.append(tool.id)
            config["enabled_tool_ids"] = enabled
            _save_config(config)
        click.echo(
            f"Enabled {_tool_kind(tool.id)} tool {tool.name} ({tool.id})"
        )

    @tool_group.command(name="disable")
    @click.argument("selector")
    def tool_disable(selector: str):
        """Remove a persisted Open WebUI tool/MCP server from CLI chats."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            tool = _resolve_tool_selector(client, selector)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc

        enabled = [
            tool_id
            for tool_id in _enabled_tool_ids(config)
            if tool_id != tool.id
        ]
        config["enabled_tool_ids"] = enabled
        _save_config(config)
        click.echo(
            f"Disabled {_tool_kind(tool.id)} tool {tool.name} ({tool.id})"
        )

    @tool_group.command(name="clear")
    def tool_clear():
        """Disable all CLI-persisted Open WebUI tools."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        config["enabled_tool_ids"] = []
        _save_config(config)
        click.echo("Disabled all CLI-persisted Open WebUI tools.")
