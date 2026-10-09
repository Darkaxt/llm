"""Open WebUI provider for LLM.

This built-in plugin keeps Open WebUI-specific auth and transport isolated from
LLM's conversation engine. It uses openwebui-sdk for email/password sign-in,
model discovery, streaming, reasoning and the Socket.IO tool execution path.
"""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import queue
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Literal
from urllib.parse import urlparse

import click
import httpx2
import llm
from llm.chat_journal import append_record as append_chat_journal_record
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
        self._lock = threading.Lock()

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
        with self._lock:
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
        with self._lock:
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
    raw = os.environ.get("LLM_OPENWEBUI_CHAT_TIMEOUT", "1200")
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


def _server_config(client: OpenWebUIClient) -> dict[str, Any]:
    payload = _owui_http_json(
        client,
        "GET",
        "/api/config",
        timeout=max(float(client.timeout), 30.0),
    )
    if not isinstance(payload, dict):
        raise llm.ModelError("Open WebUI returned an invalid /api/config response")
    return payload


def _version_triplet(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


_OPENWEBUI_MCP_CLEANUP_FIXED = (0, 9, 3)


def _server_mcp_cleanup_status(
    client: OpenWebUIClient,
) -> tuple[str | None, bool | None]:
    payload = _server_config(client)
    version = payload.get("version")
    version_text = str(version) if version is not None else None
    parsed = _version_triplet(version_text)
    if parsed is None:
        return version_text, None
    return version_text, parsed >= _OPENWEBUI_MCP_CLEANUP_FIXED


def _allow_unsafe_server_mcp() -> bool:
    return os.environ.get(
        "LLM_OPENWEBUI_ALLOW_UNSAFE_SERVER_MCP",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _guard_server_mcp_version(
    client: OpenWebUIClient,
    tool_ids: list[str],
) -> tuple[str | None, bool | None]:
    if not any(str(tool_id).startswith("server:mcp:") for tool_id in tool_ids):
        return None, None

    version, safe = _server_mcp_cleanup_status(client)
    if safe is False and not _allow_unsafe_server_mcp():
        raise llm.ModelError(
            "Refusing to route MCP through Open WebUI "
            f"{version or '<unknown>'}: versions before 0.9.3 contain a known "
            "Streamable-HTTP MCP cleanup bug that can wedge/crash the Open WebUI "
            "worker (AnyIO cancel-scope task ownership; fixed upstream by "
            "open-webui/open-webui commit adda20509 / release 0.9.3). "
            "Upgrade/backport that fix, disable the MCP tool, or set "
            "LLM_OPENWEBUI_ALLOW_UNSAFE_SERVER_MCP=1 only if this deployment "
            "already carries an equivalent backport."
        )
    return version, safe


def _model_cache(client: OpenWebUIClient) -> list[dict[str, str]]:
    return [
        {"id": model.id, "name": model.name or model.id}
        for model in client.list_models()
        if model.id
    ]


def _get_model_item(
    client: OpenWebUIClient,
    model_id: str,
) -> dict[str, Any]:
    """Return the exact model descriptor the Open WebUI browser sends."""
    payload = _owui_http_json(
        client,
        "GET",
        "/api/models",
        timeout=max(float(client.timeout), 120.0),
    )
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise llm.ModelError("Open WebUI returned an invalid /api/models response")
    for item in models:
        if isinstance(item, dict) and str(item.get("id") or "") == model_id:
            return item
    raise llm.ModelError(
        f"Open WebUI model {model_id!r} is no longer present in /api/models"
    )


def _owui_http_json(
    client: OpenWebUIClient,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    """Make an authenticated Open WebUI API request with useful errors."""
    url = f"{client.base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {client.token}",
    }
    if json_body is not None:
        headers["Content-Type"] = "application/json"

    try:
        with httpx2.Client(
            trust_env=True,
            timeout=timeout or max(float(client.timeout), 120.0),
        ) as http:
            response = http.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                data=data,
                files=files,
            )
            if not response.is_success:
                detail: Any = None
                try:
                    payload = response.json()
                    if isinstance(payload, dict):
                        detail = (
                            payload.get("detail")
                            or payload.get("message")
                            or payload.get("error")
                        )
                    if detail is None:
                        detail = payload
                except Exception:
                    detail = response.text
                raise llm.ModelError(
                    f"Open WebUI {method.upper()} {path} failed "
                    f"({response.status_code}): {detail or response.reason_phrase}"
                )
            if response.status_code == 204 or not response.content:
                return None
            try:
                return response.json()
            except Exception as exc:
                raise llm.ModelError(
                    f"Open WebUI {method.upper()} {path} returned invalid JSON"
                ) from exc
    except llm.ModelError:
        raise
    except Exception as exc:
        raise llm.ModelError(
            f"Open WebUI {method.upper()} {path} failed: {exc}"
        ) from exc


def _enabled_knowledge_ids(config: dict[str, Any]) -> list[str]:
    raw = config.get("enabled_knowledge_ids", [])
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if value]


def _knowledge_sources(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("knowledge_sources", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(knowledge_id): str(path)
        for knowledge_id, path in raw.items()
        if knowledge_id and path
    }


def _list_knowledge_bases(client: OpenWebUIClient) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = _owui_http_json(
            client,
            "GET",
            "/api/v1/knowledge/",
            params={"page": page},
        )
        if not isinstance(payload, dict):
            raise llm.ModelError("Open WebUI returned an invalid knowledge list")
        page_items = payload.get("items") or []
        if not isinstance(page_items, list):
            raise llm.ModelError("Open WebUI returned an invalid knowledge list")
        items.extend(item for item in page_items if isinstance(item, dict))
        total = payload.get("total")
        if not page_items:
            break
        if isinstance(total, int) and len(items) >= total:
            break
        page += 1
    return items


def _create_knowledge_base(
    client: OpenWebUIClient,
    name: str,
    description: str,
) -> dict[str, Any]:
    payload = _owui_http_json(
        client,
        "POST",
        "/api/v1/knowledge/create",
        json_body={
            "name": name,
            "description": description,
            "access_grants": [],
        },
        timeout=max(float(client.timeout), 300.0),
    )
    if not isinstance(payload, dict) or not payload.get("id"):
        raise llm.ModelError(
            f"Open WebUI returned an invalid knowledge base after creating {name!r}"
        )
    return payload


def _get_knowledge_by_id(
    client: OpenWebUIClient,
    knowledge_id: str,
) -> dict[str, Any]:
    payload = _owui_http_json(
        client,
        "GET",
        f"/api/v1/knowledge/{knowledge_id}",
    )
    if not isinstance(payload, dict) or not payload.get("id"):
        raise llm.ModelError(
            f"Open WebUI returned an invalid knowledge base for {knowledge_id}"
        )
    return payload


def _resolve_knowledge_selector(
    client: OpenWebUIClient,
    selector: str,
) -> dict[str, Any]:
    available = _list_knowledge_bases(client)
    if not available:
        raise click.ClickException("Open WebUI returned no knowledge bases")

    exact_id = [
        item for item in available if str(item.get("id") or "") == selector
    ]
    if exact_id:
        return exact_id[0]

    folded = selector.casefold()
    exact_name = [
        item
        for item in available
        if str(item.get("name") or "").casefold() == folded
    ]
    if len(exact_name) == 1:
        return exact_name[0]
    if len(exact_name) > 1:
        matches = ", ".join(
            f"{item.get('name')} ({item.get('id')})" for item in exact_name
        )
        raise click.ClickException(
            f"Knowledge base name is ambiguous: {matches}"
        )

    partial = [
        item
        for item in available
        if folded in str(item.get("id") or "").casefold()
        or folded in str(item.get("name") or "").casefold()
    ]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        matches = ", ".join(
            f"{item.get('name')} ({item.get('id')})" for item in partial
        )
        raise click.ClickException(
            f"Knowledge selector {selector!r} is ambiguous: {matches}"
        )
    raise click.ClickException(
        f"No Open WebUI knowledge base matches {selector!r}"
    )


def _enabled_knowledge_items(
    config: dict[str, Any],
    client: OpenWebUIClient,
) -> list[dict[str, Any]]:
    """Return collection entries shaped exactly like the browser Knowledge picker."""
    enabled_ids = _enabled_knowledge_ids(config)
    if not enabled_ids:
        return []

    available = {
        str(item.get("id")): item
        for item in _list_knowledge_bases(client)
        if isinstance(item, dict) and item.get("id")
    }
    items: list[dict[str, Any]] = []
    for knowledge_id in enabled_ids:
        knowledge = available.get(knowledge_id)
        if knowledge is None:
            raise llm.ModelError(
                f"Enabled Open WebUI knowledge base {knowledge_id} is unavailable"
            )
        items.append({"type": "collection", **knowledge})
    return items


_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SPLUNK_MACRO_RE = re.compile(r"\x60([A-Za-z0-9_.:-]+)(?:\([^)]*\))?\x60")


def _knowledge_search_files(
    client: OpenWebUIClient,
    knowledge_id: str,
    query: str,
    *,
    include_content: bool = False,
) -> list[dict[str, Any]]:
    payload = _owui_http_json(
        client,
        "GET",
        f"/api/v1/knowledge/{knowledge_id}/files",
        params={
            "query": query,
            "include_content": "true" if include_content else "false",
            "page": 1,
        },
        timeout=max(float(client.timeout), 120.0),
    )
    if not isinstance(payload, dict):
        raise llm.ModelError(
            f"Open WebUI returned an invalid file search for knowledge {knowledge_id}"
        )
    items = payload.get("items") or []
    return [item for item in items if isinstance(item, dict)]


def _knowledge_exact_file(
    client: OpenWebUIClient,
    knowledge_id: str,
    filename: str,
) -> dict[str, Any] | None:
    matches = _knowledge_search_files(client, knowledge_id, filename)
    exact = [
        item
        for item in matches
        if str(item.get("filename") or "").casefold() == filename.casefold()
    ]
    if not exact:
        return None
    if len(exact) > 1:
        ids = ", ".join(str(item.get("id") or "?") for item in exact[:5])
        raise llm.ModelError(
            f"Knowledge base contains multiple files named {filename!r}: {ids}"
        )
    return exact[0]


def _knowledge_file_text(
    client: OpenWebUIClient,
    file_id: str,
) -> str:
    payload = _owui_http_json(
        client,
        "GET",
        f"/api/v1/files/{file_id}/data/content",
        timeout=max(float(client.timeout), 120.0),
    )
    if not isinstance(payload, dict):
        raise llm.ModelError(
            f"Open WebUI returned invalid content for knowledge file {file_id}"
        )
    return str(payload.get("content") or "")


def _json_contains_needles(value: Any, needles: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            _json_contains_needles(key, needles)
            or _json_contains_needles(item, needles)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_json_contains_needles(item, needles) for item in value)
    text = str(value)
    return any(needle in text for needle in needles)


def _json_matching_fragments(
    value: Any,
    needles: set[str],
    *,
    max_matches: int = 20,
) -> list[Any]:
    matches: list[Any] = []

    def visit(node: Any) -> None:
        if len(matches) >= max_matches:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                if len(matches) >= max_matches:
                    return
                if _json_contains_needles(key, needles) or _json_contains_needles(
                    child, needles
                ):
                    matches.append({key: child})
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                if len(matches) >= max_matches:
                    return
                if _json_contains_needles(child, needles):
                    matches.append(child)
                elif isinstance(child, (dict, list)):
                    visit(child)

    if _json_contains_needles(value, needles):
        if isinstance(value, dict):
            visit(value)
        elif isinstance(value, list):
            visit(value)
        else:
            matches.append(value)
    return matches[:max_matches]


def _extract_macro_subset(macros: Any, macro_names: set[str]) -> Any:
    if not macro_names:
        return {}

    wanted = {name.casefold() for name in macro_names}
    selected: dict[str, Any] = {}

    def visit(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                full_path = f"{path}.{key_text}" if path else key_text
                if key_text.casefold() in wanted:
                    selected[full_path] = value
                    continue
                if isinstance(value, dict):
                    name = value.get("name")
                    if isinstance(name, str) and name.casefold() in wanted:
                        selected[full_path] = value
                        continue
                visit(value, full_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, dict):
                    name = value.get("name")
                    if isinstance(name, str) and name.casefold() in wanted:
                        selected[f"{path}[{index}]"] = value
                        continue
                visit(value, f"{path}[{index}]")

    visit(macros)
    return selected


def _conversation_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif content is not None:
            chunks.append(json.dumps(content, ensure_ascii=False, default=str))
    return "\n".join(chunks)


def _resolve_knowledge_context(
    client: OpenWebUIClient,
    knowledge_items: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    *,
    on_status: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve a compact deterministic context from persistent Knowledge.

    This avoids Open WebUI's automatic collection RAG and avoids exposing its
    whole builtin-tool catalogue to providers with strict/limited native tool
    handling. It intentionally extracts only authoritative workflow files and
    exact identifier-bearing TIDE evidence.
    """
    if not knowledge_items:
        return messages, {}

    conversation_text = _conversation_text(messages)
    uuids = set(_UUID_RE.findall(conversation_text))
    context_sections: list[str] = []
    resolution_meta: dict[str, Any] = {
        "knowledge_bases": [],
        "uuids": sorted(uuids),
        "files": [],
    }

    for knowledge in knowledge_items:
        knowledge_id = str(knowledge.get("id") or "")
        knowledge_name = str(knowledge.get("name") or knowledge_id)
        if not knowledge_id:
            continue
        resolution_meta["knowledge_bases"].append(
            {"id": knowledge_id, "name": knowledge_name}
        )

        if on_status is not None:
            on_status(f"knowledge resolve · {knowledge_name}")

        fetched: dict[str, tuple[str, str]] = {}
        for filename in (
            "SKILL.md",
            "SPEC.md",
            "rule-index.json",
            "splunk-rules.jsonl",
            "macros.json",
        ):
            item = _knowledge_exact_file(client, knowledge_id, filename)
            if item is None:
                continue
            file_id = str(item.get("id") or "")
            if not file_id:
                continue
            text = _knowledge_file_text(client, file_id)
            fetched[filename] = (file_id, text)
            resolution_meta["files"].append(
                {
                    "knowledge_id": knowledge_id,
                    "id": file_id,
                    "filename": filename,
                    "chars": len(text),
                }
            )

        section_lines = [
            f'<persistent_knowledge name="{knowledge_name}" id="{knowledge_id}">'
        ]

        for filename in ("SKILL.md", "SPEC.md"):
            if filename in fetched:
                section_lines.extend(
                    [
                        f'<file name="{filename}">',
                        fetched[filename][1],
                        "</file>",
                    ]
                )

        rule_fragments: list[Any] = []
        if uuids and "rule-index.json" in fetched:
            raw_index = fetched["rule-index.json"][1]
            try:
                parsed_index = json.loads(raw_index)
                rule_fragments = _json_matching_fragments(parsed_index, uuids)
            except Exception:
                matching_lines = [
                    line
                    for line in raw_index.splitlines()
                    if any(uuid_value in line for uuid_value in uuids)
                ][:20]
                rule_fragments = matching_lines
            if rule_fragments:
                section_lines.extend(
                    [
                        '<exact_evidence source="rule-index.json">',
                        json.dumps(
                            rule_fragments,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        "</exact_evidence>",
                    ]
                )

        rule_records: list[Any] = []
        if uuids and "splunk-rules.jsonl" in fetched:
            for line in fetched["splunk-rules.jsonl"][1].splitlines():
                if not any(uuid_value in line for uuid_value in uuids):
                    continue
                try:
                    rule_records.append(json.loads(line))
                except Exception:
                    rule_records.append(line)
                if len(rule_records) >= 10:
                    break
            if rule_records:
                section_lines.extend(
                    [
                        '<exact_evidence source="splunk-rules.jsonl">',
                        json.dumps(
                            rule_records,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        "</exact_evidence>",
                    ]
                )

        macro_names: set[str] = set()
        evidence_text = json.dumps(
            [rule_fragments, rule_records],
            ensure_ascii=False,
            default=str,
        )
        macro_names.update(_SPLUNK_MACRO_RE.findall(evidence_text))

        if "macros.json" in fetched and macro_names:
            raw_macros = fetched["macros.json"][1]
            try:
                parsed_macros = json.loads(raw_macros)
                macro_subset = _extract_macro_subset(parsed_macros, macro_names)
            except Exception:
                macro_subset = {
                    name: [
                        line
                        for line in raw_macros.splitlines()
                        if name in line
                    ][:10]
                    for name in sorted(macro_names)
                }
            if macro_subset:
                section_lines.extend(
                    [
                        '<exact_evidence source="macros.json">',
                        json.dumps(
                            macro_subset,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        "</exact_evidence>",
                    ]
                )

        section_lines.extend(
            [
                "<instructions>",
                "The material above was resolved deterministically from the "
                "persistent Open WebUI Knowledge Base before this model call.",
                "Treat SKILL.md and SPEC.md as authoritative when present.",
                "Exact evidence blocks contain only records matching identifiers "
                "from the conversation. Do not assume omitted KB records are absent.",
                "</instructions>",
                "</persistent_knowledge>",
            ]
        )
        context_sections.append("\n".join(section_lines))

    if not context_sections:
        return messages, resolution_meta

    resolved_context = "\n\n".join(context_sections)
    scoped = [dict(message) for message in messages]
    if scoped and scoped[0].get("role") == "system":
        existing = str(scoped[0].get("content") or "")
        scoped[0]["content"] = (
            existing + "\n\n" + resolved_context
            if existing
            else resolved_context
        )
    else:
        scoped.insert(0, {"role": "system", "content": resolved_context})

    resolution_meta["context_chars"] = len(resolved_context)
    resolution_meta["macro_names"] = sorted(macro_names) if 'macro_names' in locals() else []
    return scoped, resolution_meta


def _enabled_tool_ids(config: dict[str, Any]) -> list[str]:
    raw = config.get("enabled_tool_ids", [])
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if value]


_KB_ARCHIVE_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_KB_ARCHIVE_MAX_TOTAL_BYTES = 512 * 1024 * 1024


def _safe_archive_member_path(name: str) -> PurePosixPath | None:
    normalized = str(name).replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if not path.name:
        return None
    if any(part in {"", ".", ".."} for part in path.parts):
        if ".." in path.parts:
            raise click.ClickException(
                f"Archive member uses an unsafe relative path: {name!r}"
            )
    if any(part.startswith(".") for part in path.parts):
        return None
    return path


def _zip_knowledge_members(path: Path) -> list[tuple[PurePosixPath, bytes]]:
    members: list[tuple[PurePosixPath, bytes]] = []
    total = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_path = _safe_archive_member_path(info.filename)
                if member_path is None:
                    continue
                if info.file_size > _KB_ARCHIVE_MAX_MEMBER_BYTES:
                    raise click.ClickException(
                        f"Archive member {info.filename!r} is too large "
                        f"({info.file_size} bytes)"
                    )
                total += int(info.file_size)
                if total > _KB_ARCHIVE_MAX_TOTAL_BYTES:
                    raise click.ClickException(
                        f"Expanded archive {path} exceeds "
                        f"{_KB_ARCHIVE_MAX_TOTAL_BYTES} bytes"
                    )
                members.append((member_path, archive.read(info)))
    except zipfile.BadZipFile as exc:
        raise click.ClickException(f"Invalid ZIP archive {path}: {exc}") from exc
    return members


def _sevenzip_knowledge_members(path: Path) -> list[tuple[PurePosixPath, bytes]]:
    try:
        import py7zr
        from py7zr.io import BytesIOFactory
    except ImportError as exc:
        raise click.ClickException(
            "7z knowledge sync requires py7zr; reinstall with "
            "python -m pip install -e ."
        ) from exc

    try:
        with py7zr.SevenZipFile(path, mode="r") as archive:
            if archive.needs_password():
                raise click.ClickException(
                    f"7z archive {path} is password-protected"
                )
            infos = [info for info in archive.list() if not info.is_directory]
            selected: list[tuple[str, PurePosixPath]] = []
            total = 0
            for info in infos:
                member_path = _safe_archive_member_path(info.filename)
                if member_path is None:
                    continue
                size = int(info.uncompressed or 0)
                if size > _KB_ARCHIVE_MAX_MEMBER_BYTES:
                    raise click.ClickException(
                        f"Archive member {info.filename!r} is too large "
                        f"({size} bytes)"
                    )
                total += size
                if total > _KB_ARCHIVE_MAX_TOTAL_BYTES:
                    raise click.ClickException(
                        f"Expanded archive {path} exceeds "
                        f"{_KB_ARCHIVE_MAX_TOTAL_BYTES} bytes"
                    )
                selected.append((info.filename, member_path))

            if not selected:
                return []

            factory = BytesIOFactory(_KB_ARCHIVE_MAX_MEMBER_BYTES + 1)
            archive.extract(
                targets=[raw_name for raw_name, _ in selected],
                factory=factory,
            )

            members: list[tuple[PurePosixPath, bytes]] = []
            for raw_name, member_path in selected:
                product = factory.get(raw_name)
                product.seek(0)
                data = product.read()
                if len(data) > _KB_ARCHIVE_MAX_MEMBER_BYTES:
                    raise click.ClickException(
                        f"Archive member {raw_name!r} exceeded the extraction limit"
                    )
                members.append((member_path, data))
            return members
    except click.ClickException:
        raise
    except py7zr.Bad7zFile as exc:
        raise click.ClickException(f"Invalid 7z archive {path}: {exc}") from exc
    except Exception as exc:
        raise click.ClickException(
            f"Failed to read 7z archive {path}: {exc}"
        ) from exc


def _knowledge_archive_members(
    path: Path,
) -> list[tuple[PurePosixPath, bytes]] | None:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return _zip_knowledge_members(path)
    if suffix == ".7z":
        return _sevenzip_knowledge_members(path)
    return None


def _local_knowledge_manifest(root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise click.ClickException(f"Knowledge source folder does not exist: {root}")
    if not root.is_dir():
        raise click.ClickException(f"Knowledge source is not a folder: {root}")

    manifest: list[dict[str, Any]] = []
    seen_virtual_paths: dict[tuple[str, str], str] = {}

    def add_entry(
        *,
        virtual_path: PurePosixPath,
        content: bytes | None,
        local_path: Path,
        source_label: str,
    ) -> None:
        parent = virtual_path.parent.as_posix()
        if parent == ".":
            parent = ""
        key = (parent, virtual_path.name)
        previous = seen_virtual_paths.get(key)
        if previous is not None:
            display = f"{parent}/{virtual_path.name}" if parent else virtual_path.name
            raise click.ClickException(
                f"Knowledge source contains duplicate virtual path {display!r}: "
                f"{previous} and {source_label}"
            )
        seen_virtual_paths[key] = source_label

        if content is None:
            digest = hashlib.sha256()
            size = 0
            with local_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
            checksum = digest.hexdigest()
        else:
            size = len(content)
            checksum = hashlib.sha256(content).hexdigest()

        entry: dict[str, Any] = {
            "filename": virtual_path.name,
            "path": parent,
            "checksum": checksum,
            "size": size,
            "_local_path": str(local_path),
        }
        if content is not None:
            entry["_content"] = content
            entry["_archive_source"] = source_label
        manifest.append(entry)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".")
            and not (Path(dirpath) / name).is_symlink()
        )
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            full_path = Path(dirpath) / filename
            if full_path.is_symlink() or not full_path.is_file():
                continue

            relative = full_path.relative_to(root)
            archive_members = _knowledge_archive_members(full_path)
            if archive_members is None:
                add_entry(
                    virtual_path=PurePosixPath(relative.as_posix()),
                    content=None,
                    local_path=full_path,
                    source_label=str(relative),
                )
                continue

            archive_parent = PurePosixPath(relative.parent.as_posix())
            for member_path, data in archive_members:
                virtual_path = (
                    archive_parent / member_path
                    if archive_parent.as_posix() not in {"", "."}
                    else member_path
                )
                add_entry(
                    virtual_path=virtual_path,
                    content=data,
                    local_path=full_path,
                    source_label=f"{relative}!/{member_path.as_posix()}",
                )

    manifest.sort(
        key=lambda item: (
            str(item["path"]).casefold(),
            str(item["filename"]).casefold(),
        )
    )
    return manifest



def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "application/x-ndjson"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type or "application/octet-stream"


def _list_user_files(client: OpenWebUIClient) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = _owui_http_json(
            client,
            "GET",
            "/api/v1/files/",
            params={"page": page, "content": "false"},
        )
        if not isinstance(payload, dict):
            raise llm.ModelError("Open WebUI returned an invalid file list")
        page_items = payload.get("items") or []
        if not isinstance(page_items, list):
            raise llm.ModelError("Open WebUI returned an invalid file list")
        items.extend(item for item in page_items if isinstance(item, dict))
        total = payload.get("total")
        if not page_items:
            break
        if isinstance(total, int) and len(items) >= total:
            break
        page += 1
    return items


def _knowledge_file_candidates(
    client: OpenWebUIClient,
    knowledge_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Index prior uploads for this KB by the raw local-sync SHA-256."""
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for item in _list_user_files(client):
        meta = item.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        sync_meta = meta.get("data") or {}
        if not isinstance(sync_meta, dict):
            continue
        if str(sync_meta.get("knowledge_id") or "") != knowledge_id:
            continue
        file_hash = str(meta.get("file_hash") or "")
        if not file_hash:
            continue
        by_hash.setdefault(file_hash, []).append(item)

    # Prefer the oldest upload: if a later retry failed with duplicate content,
    # the older file ID is the one that already owns the KB vector chunks.
    for candidates in by_hash.values():
        candidates.sort(
            key=lambda item: (
                int(item.get("created_at") or 0),
                str(item.get("id") or ""),
            )
        )
    return by_hash


def _create_knowledge_directory(
    client: OpenWebUIClient,
    knowledge_id: str,
    directory_path: str,
    directory_ids: dict[str, str],
) -> str:
    segments = [segment for segment in directory_path.split("/") if segment]
    current_path = ""
    parent_id: str | None = None
    for segment in segments:
        current_path = (
            f"{current_path}/{segment}" if current_path else segment
        )
        existing = directory_ids.get(current_path)
        if existing:
            parent_id = existing
            continue
        payload = _owui_http_json(
            client,
            "POST",
            f"/api/v1/knowledge/{knowledge_id}/dirs/create",
            json_body={
                "name": segment,
                **({"parent_id": parent_id} if parent_id else {}),
            },
        )
        if not isinstance(payload, dict) or not payload.get("id"):
            raise llm.ModelError(
                f"Open WebUI failed to create knowledge directory "
                f"{current_path!r}"
            )
        parent_id = str(payload["id"])
        directory_ids[current_path] = parent_id
    if not parent_id:
        raise llm.ModelError(
            f"Could not resolve knowledge directory {directory_path!r}"
        )
    return parent_id


def _upload_knowledge_sync_file(
    client: OpenWebUIClient,
    *,
    knowledge_id: str,
    entry: dict[str, Any],
    directory_id: str | None,
    ordinal: int,
    total: int,
    on_status: Callable[[str], None] | None = None,
    interactive_status: bool = False,
) -> dict[str, Any]:
    path = Path(str(entry["_local_path"]))
    if "_content" in entry:
        content = bytes(entry["_content"])
    else:
        try:
            content = path.read_bytes()
        except Exception as exc:
            raise llm.ModelError(
                f"Could not read knowledge source file {path}: {exc}"
            ) from exc

    metadata = {
        "knowledge_id": knowledge_id,
        "file_hash": entry["checksum"],
        "directory_id": directory_id,
    }
    started = time.monotonic()
    _render_file_progress(
        filename=(
            f"{entry['path']}/{entry['filename']}"
            if entry["path"]
            else entry["filename"]
        ),
        status="uploading to knowledge",
        started=started,
        tick=0,
        ordinal=ordinal,
        total=total,
        on_status=on_status,
    )

    with httpx2.Client(
        trust_env=True,
        timeout=max(float(client.timeout), 120.0),
    ) as http:
        try:
            result = http.post(
                f"{client.base_url}/api/v1/files/",
                params={
                    "process": "true",
                    "process_in_background": "true",
                },
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {client.token}",
                },
                data={"metadata": json.dumps(metadata)},
                files={
                    "file": (
                        entry["filename"],
                        content,
                        _content_type_for_path(Path(str(entry["filename"]))),
                    )
                },
            )
            result.raise_for_status()
            uploaded = result.json()
        except Exception as exc:
            raise llm.ModelError(
                f"Open WebUI failed to upload knowledge file {path}: {exc}"
            ) from exc

        if not isinstance(uploaded, dict) or not uploaded.get("id"):
            raise llm.ModelError(
                f"Open WebUI returned an invalid knowledge upload "
                f"response for {path}"
            )

        _wait_for_file_processing(
            http,
            client,
            str(uploaded["id"]),
            str(entry["filename"]),
            ordinal=ordinal,
            total=total,
            started=started,
            on_status=on_status,
            interactive_status=interactive_status,
        )
        return uploaded


def _reset_knowledge_base(
    client: OpenWebUIClient,
    knowledge_id: str,
) -> dict[str, Any]:
    payload = _owui_http_json(
        client,
        "POST",
        f"/api/v1/knowledge/{knowledge_id}/reset",
        params={"include_directories": "true"},
        timeout=max(float(client.timeout), 600.0),
    )
    if not isinstance(payload, dict) or not payload.get("id"):
        raise llm.ModelError(
            f"Open WebUI returned an invalid reset response for {knowledge_id}"
        )
    return payload


def _sync_knowledge_folder(
    client: OpenWebUIClient,
    knowledge: dict[str, Any],
    root: Path,
    *,
    on_status: Callable[[str], None] | None = None,
    interactive_status: bool = False,
    detect_interrupted: bool = True,
) -> dict[str, int]:
    knowledge_id = str(knowledge["id"])
    root = root.expanduser().resolve()

    if on_status is not None:
        on_status(f"knowledge sync · hashing {root}")
    manifest = _local_knowledge_manifest(root)
    if not manifest:
        raise click.ClickException(
            f"Knowledge source folder contains no visible files: {root}. "
            "Refusing to sync an empty folder."
        )
    wire_manifest = [
        {
            key: entry[key]
            for key in ("filename", "path", "checksum", "size")
        }
        for entry in manifest
    ]

    if on_status is not None:
        on_status(
            f"knowledge sync · comparing {len(wire_manifest)} local files"
        )
    diff = _owui_http_json(
        client,
        "POST",
        f"/api/v1/knowledge/{knowledge_id}/sync/diff",
        json_body={"manifest": wire_manifest},
        timeout=max(float(client.timeout), 300.0),
    )
    if not isinstance(diff, dict):
        raise llm.ModelError("Open WebUI returned an invalid knowledge sync diff")

    added = diff.get("added") or []
    modified = diff.get("modified") or []
    deleted = diff.get("deleted") or []
    rmdir = diff.get("rmdir") or []
    mkdir = diff.get("mkdir") or []
    directory_ids = {
        str(path): str(directory_id)
        for path, directory_id in (diff.get("directory_map") or {}).items()
        if path and directory_id
    }

    wanted = {
        (str(item.get("path") or ""), str(item.get("filename") or ""))
        for item in [*added, *modified]
        if isinstance(item, dict)
    }
    files_to_upload = [
        entry
        for entry in manifest
        if (entry["path"], entry["filename"]) in wanted
    ]

    # Detect an interrupted population before making any further server-side
    # changes. A file that sync/diff says is missing but that already has a
    # prior upload record for this KB is an inconsistent/orphaned state.
    if detect_interrupted and files_to_upload:
        candidates_by_hash = _knowledge_file_candidates(
            client,
            knowledge_id,
        )
        interrupted: list[str] = []
        for entry in files_to_upload:
            matching = []
            for candidate in candidates_by_hash.get(str(entry["checksum"]), []):
                meta = candidate.get("meta") or {}
                meta_name = meta.get("name") if isinstance(meta, dict) else None
                candidate_name = str(
                    meta_name or candidate.get("filename") or ""
                )
                if candidate_name == str(entry["filename"]):
                    matching.append(candidate)
            if matching:
                display = (
                    f"{entry['path']}/{entry['filename']}"
                    if entry["path"]
                    else str(entry["filename"])
                )
                interrupted.append(display)

        if interrupted:
            preview = ", ".join(interrupted[:5])
            if len(interrupted) > 5:
                preview += f", … (+{len(interrupted) - 5} more)"
            raise click.ClickException(
                "Interrupted knowledge population detected: "
                f"{len(interrupted)} file(s) are missing from the KB but matching "
                "prior Open WebUI uploads already exist. No KB changes were made. "
                "Run: llm openwebui knowledge rebuild "
                f"\"{knowledge.get('name') or knowledge_id}\". "
                f"Examples: {preview}"
            )

    stale_ids = [
        str(item.get("file_id"))
        for item in deleted
        if isinstance(item, dict) and item.get("file_id")
    ] + [
        str(item.get("stale_file_id"))
        for item in modified
        if isinstance(item, dict) and item.get("stale_file_id")
    ]

    if stale_ids or rmdir:
        if on_status is not None:
            on_status(
                f"knowledge sync · removing {len(stale_ids)} stale files"
            )
        _owui_http_json(
            client,
            "POST",
            f"/api/v1/knowledge/{knowledge_id}/sync/cleanup",
            json_body={
                "file_ids": stale_ids,
                "dir_ids": [str(value) for value in rmdir],
            },
            timeout=max(float(client.timeout), 300.0),
        )

    for directory_path in sorted(
        (str(path) for path in mkdir if path),
        key=lambda value: (value.count("/"), value.casefold()),
    ):
        _create_knowledge_directory(
            client,
            knowledge_id,
            directory_path,
            directory_ids,
        )

    work_items: list[tuple[int, dict[str, Any], str | None]] = []
    for ordinal, entry in enumerate(files_to_upload, start=1):
        directory_id = (
            directory_ids.get(entry["path"])
            if entry["path"]
            else None
        )
        if entry["path"] and not directory_id:
            directory_id = _create_knowledge_directory(
                client,
                knowledge_id,
                entry["path"],
                directory_ids,
            )
        work_items.append((ordinal, entry, directory_id))

    if not work_items:
        return {
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
            "unmodified": int(diff.get("unmodified_count") or 0),
            "uploaded": 0,
            "reused": 0,
        }


    concurrency = min(_knowledge_sync_concurrency(), len(work_items))
    progress_lock = threading.Lock()
    completed_count = 0
    reused = 0
    uploaded_count = 0
    active: dict[str, str] = {}

    def display_path(entry: dict[str, Any]) -> str:
        return (
            f"{entry['path']}/{entry['filename']}"
            if entry["path"]
            else str(entry["filename"])
        )

    def emit_progress(latest: str | None = None) -> None:
        if on_status is None:
            return
        with progress_lock:
            active_count = len(active)
            done = completed_count
            suffix = f" · {latest}" if latest else ""
            message = (
                f"knowledge sync · {done}/{len(work_items)} complete · "
                f"{active_count} in flight · concurrency {concurrency}{suffix}"
            )
        on_status(message)

    def process_item(
        ordinal: int,
        entry: dict[str, Any],
        directory_id: str | None,
    ) -> str:
        nonlocal completed_count, reused, uploaded_count
        path_label = display_path(entry)

        with progress_lock:
            active[path_label] = "starting"
        emit_progress(path_label)

        def worker_status(message: str) -> None:
            # Preserve only the latest per-file state; expose aggregate sync
            # progress instead of allowing concurrent workers to fight over
            # the single terminal status line.
            with progress_lock:
                active[path_label] = message
            emit_progress(path_label)

        try:
            _upload_knowledge_sync_file(
                client,
                knowledge_id=knowledge_id,
                entry=entry,
                directory_id=directory_id,
                ordinal=ordinal,
                total=len(work_items),
                on_status=worker_status,
                interactive_status=False,
            )
            outcome = "uploaded"

            with progress_lock:
                uploaded_count += 1
                completed_count += 1
                active.pop(path_label, None)
            emit_progress(f"{path_label} · {outcome}")
            return outcome
        except Exception:
            with progress_lock:
                active.pop(path_label, None)
            emit_progress(f"{path_label} · failed")
            raise

    if on_status is not None:
        on_status(
            f"knowledge sync · starting {len(work_items)} files · "
            f"concurrency {concurrency}"
        )

    futures = []
    with ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="owui-kb",
    ) as executor:
        for ordinal, entry, directory_id in work_items:
            futures.append(
                executor.submit(
                    process_item,
                    ordinal,
                    entry,
                    directory_id,
                )
            )

        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            # Stop work that has not started yet. Running requests are allowed
            # to unwind normally so httpx/background processing is not left in
            # an indeterminate local state.
            for future in futures:
                future.cancel()
            raise

    return {
        "added": len(added),
        "modified": len(modified),
        "deleted": len(deleted),
        "unmodified": int(diff.get("unmodified_count") or 0),
        "uploaded": uploaded_count,
        "reused": reused,
    }


def _journal_file_reference(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "type",
        "id",
        "name",
        "description",
        "context",
        "collection_name",
        "content_type",
        "size",
        "status",
        "url",
    )
    return {
        key: item.get(key)
        for key in keys
        if item.get(key) is not None
    }


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
        from py7zr.io import BytesIOFactory
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


def _knowledge_sync_concurrency() -> int:
    raw = os.environ.get("LLM_OPENWEBUI_KB_CONCURRENCY", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise llm.ModelError(
            "LLM_OPENWEBUI_KB_CONCURRENCY must be an integer"
        ) from exc
    if value < 1:
        raise llm.ModelError(
            "LLM_OPENWEBUI_KB_CONCURRENCY must be at least 1"
        )
    # Keep accidental values from overwhelming the Open WebUI deployment.
    return min(value, 16)


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
    timeout_seconds: float | None = None,
) -> None:
    """Wait for Open WebUI's background extraction/RAG processing.

    The browser uses /process/status after the upload POST instead of keeping
    the upload request open. Polling the non-streaming status endpoint gives us
    the same semantics without tying a long-running SSE connection to the CLI.
    """
    started = started if started is not None else time.monotonic()
    processing_timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else _file_processing_timeout()
    )
    deadline = started + processing_timeout
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
                f"after {processing_timeout:g}s"
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
        openwebui_mcp_transport: Literal["sessionless_native", "background_legacy"] = (
            "sessionless_native"
        )
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
        provider_run_id = (
            f"{conversation.id}:{time.time_ns()}"
            if conversation is not None
            else f"transient:{time.time_ns()}"
        )

        def journal_provider(record: dict[str, Any]) -> None:
            if conversation is not None:
                append_chat_journal_record(
                    conversation.id,
                    {
                        "provider": "openwebui",
                        "provider_run_id": provider_run_id,
                        **record,
                    },
                )

        def prepare_status(line: str) -> None:
            status_activity.append(line)
            journal_provider(
                {
                    "type": "provider_status",
                    "status": line,
                }
            )
            status_bar.update(line)

        try:
            messages, attached_files = _prepare_openwebui_request(
                prompt,
                client,
                on_status=prepare_status,
                interactive_status=status_bar.enabled,
            )
            knowledge_items = _enabled_knowledge_items(config, client)
            knowledge_resolution: dict[str, Any] = {}
            if knowledge_items:
                messages, knowledge_resolution = _resolve_knowledge_context(
                    client,
                    knowledge_items,
                    messages,
                    on_status=prepare_status,
                )
                prepare_status(
                    "knowledge resolved · "
                    + " · ".join(
                        str(item.get("name") or item.get("id"))
                        for item in knowledge_items
                    )
                )
        except Exception:
            status_bar.clear()
            raise

        try:
            model_item = _get_model_item(client, self.remote_model_id)
            # CLI tool selection is authoritative. Do not merge model-attached
            # defaults from info.meta.toolIds: tool clear / tool disable must
            # actually remove those tools for CLI chats.
            tool_ids = (
                _enabled_tool_ids(config)
                if prompt.options.openwebui_tools
                else []
            )
            server_version, server_mcp_safe = _guard_server_mcp_version(
                client,
                tool_ids,
            )
        except (APIError, AuthError) as exc:
            raise llm.ModelError(str(exc)) from exc

        journal_provider(
            {
                "type": "provider_request",
                "model": self.remote_model_id,
                "messages": messages,
                "tool_ids": tool_ids,
                "files": [
                    _journal_file_reference(item)
                    for item in attached_files
                    if isinstance(item, dict)
                ],
                "knowledge_scope": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                    }
                    for item in knowledge_items
                ],
                "knowledge_resolution": knowledge_resolution,
                "options": {
                    "temperature": prompt.options.temperature,
                    "openwebui_attachment_context": (
                        prompt.options.openwebui_attachment_context
                    ),
                    "openwebui_tools": prompt.options.openwebui_tools,
                    "sessionless_server_tools": (
                        bool(knowledge_items)
                        and prompt.options.openwebui_mcp_transport == "sessionless_native"
                    ),
                    "openwebui_mcp_transport": prompt.options.openwebui_mcp_transport,
                },
                "server": {
                    "version": server_version,
                    "mcp_cleanup_safe": server_mcp_safe,
                },
                "model_item": {
                    "id": model_item.get("id"),
                    "name": model_item.get("name"),
                    "owned_by": model_item.get("owned_by"),
                    "direct": model_item.get("direct"),
                    "info": model_item.get("info"),
                },
            }
        )

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        cancel_requested = threading.Event()
        worker_finished = threading.Event()
        tool_activity: list[str] = []
        remote_execution: dict[str, Any] = {}

        def on_text(fragment: str) -> None:
            events.put(("text", fragment))

        def on_reasoning(fragment: str) -> None:
            events.put(("reasoning", fragment))

        def on_tool(line: str) -> None:
            tool_activity.append(line)
            journal_provider(
                {
                    "type": "provider_tool",
                    "tool": line,
                }
            )
            events.put(("tool", line))

        def on_status(line: str) -> None:
            status_activity.append(line)
            journal_provider(
                {
                    "type": "provider_status",
                    "status": line,
                }
            )
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
                            model_item=model_item,
                            messages=messages,
                            tool_ids=tool_ids,
                            files=attached_files,
                            params={
                                **(
                                    {"temperature": prompt.options.temperature}
                                    if prompt.options.temperature is not None
                                    else {}
                                ),
                                **(
                                    {"function_calling": "legacy"}
                                    if prompt.options.openwebui_mcp_transport
                                    == "background_legacy"
                                    else {}
                                ),
                            },
                            timeout=client.timeout,
                            sessionless_server_tools=(
                                bool(knowledge_items)
                                and prompt.options.openwebui_mcp_transport
                                == "sessionless_native"
                            ),
                            stop_requested=cancel_requested,
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
                journal_provider(
                    {
                        "type": "provider_error",
                        "error": str(exc),
                        "error_class": type(exc).__name__,
                    }
                )
                events.put(("error", exc))
            finally:
                worker_finished.set()

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
                        context_bits = []
                        if status_activity:
                            context_bits.append(
                                f"last status: {status_activity[-1]}"
                            )
                        if remote_execution.get("remote_chat_id"):
                            context_bits.append(
                                f"remote chat: {remote_execution['remote_chat_id']}"
                            )
                        task_ids = remote_execution.get("remote_task_ids") or []
                        if task_ids:
                            context_bits.append(
                                "remote task(s): " + ",".join(task_ids)
                            )
                        suffix = (
                            " [" + "; ".join(context_bits) + "]"
                            if context_bits
                            else ""
                        )
                        raise llm.ModelError(str(payload) + suffix) from payload
                    raise payload
                elif kind == "done":
                    status_bar.clear()
                    result = payload
                    response.response_json = {
                        "provider": "openwebui",
                        "provider_run_id": provider_run_id,
                        "remote_model": self.remote_model_id,
                        "reasoning": result.reasoning,
                        "tool_calls": result.tool_calls,
                        "tool_activity": tool_activity,
                        "status_activity": status_activity,
                        "raw_content": getattr(result, "raw_content", ""),
                        **remote_execution,
                    }
                    break
        finally:
            if not worker_finished.is_set():
                cancel_requested.set()
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
            server_version, server_mcp_safe = _server_mcp_cleanup_status(client)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc
        except llm.ModelError as exc:
            raise click.ClickException(str(exc)) from exc

        output = {
            "user": config.get("user", {}),
            "server": {
                "version": server_version,
                "mcp_cleanup_fix": (
                    "present"
                    if server_mcp_safe is True
                    else "missing"
                    if server_mcp_safe is False
                    else "unknown"
                ),
                "minimum_safe_version": "0.9.3",
            },
        }
        click.echo(json.dumps(output, indent=2, ensure_ascii=False))

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


    @openwebui_group.group(name="knowledge")
    def knowledge_group():
        """Manage reusable Open WebUI knowledge bases."""

    @knowledge_group.command(name="list")
    def knowledge_list():
        """List knowledge bases visible to this user."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            available = _list_knowledge_bases(client)
        except llm.ModelError as exc:
            raise click.ClickException(str(exc)) from exc

        enabled = set(_enabled_knowledge_ids(config))
        sources = _knowledge_sources(config)
        seen: set[str] = set()
        if not available and not enabled:
            click.echo("No Open WebUI knowledge bases are visible to this user.")
            return

        for item in available:
            knowledge_id = str(item.get("id") or "")
            if not knowledge_id:
                continue
            seen.add(knowledge_id)
            marker = "*" if knowledge_id in enabled else " "
            name = str(item.get("name") or knowledge_id)
            source = sources.get(knowledge_id, "")
            writable = item.get("write_access")
            access = "write" if writable is not False else "read"
            click.echo(
                f"{marker}\t{name}\t{knowledge_id}\t{access}"
                + (f"\t{source}" if source else "")
            )

        for knowledge_id in sorted(enabled - seen):
            source = sources.get(knowledge_id, "")
            click.echo(
                f"*\t<unavailable>\t{knowledge_id}\tstale"
                + (f"\t{source}" if source else "")
            )

    @knowledge_group.command(name="create")
    @click.argument("name")
    @click.option(
        "--description",
        default="Managed by llm Open WebUI CLI.",
        show_default=True,
        help="Knowledge base description.",
    )
    @click.option(
        "--enable",
        is_flag=True,
        help="Enable this knowledge base for future CLI chats.",
    )
    def knowledge_create(name: str, description: str, enable: bool):
        """Create a persistent Open WebUI knowledge base."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            knowledge = _create_knowledge_base(client, name, description)
        except llm.ModelError as exc:
            raise click.ClickException(str(exc)) from exc

        knowledge_id = str(knowledge["id"])
        if enable:
            enabled = _enabled_knowledge_ids(config)
            if knowledge_id not in enabled:
                enabled.append(knowledge_id)
                config["enabled_knowledge_ids"] = enabled
                _save_config(config)
        click.echo(
            f"Created knowledge base {knowledge.get('name') or name} "
            f"({knowledge_id})"
            + (" and enabled it for CLI chats." if enable else ".")
        )

    @knowledge_group.command(name="enable")
    @click.argument("selector")
    def knowledge_enable(selector: str):
        """Enable a persistent knowledge base for every Open WebUI CLI chat."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            knowledge = _resolve_knowledge_selector(client, selector)
        except llm.ModelError as exc:
            raise click.ClickException(str(exc)) from exc

        knowledge_id = str(knowledge["id"])
        enabled = _enabled_knowledge_ids(config)
        if knowledge_id not in enabled:
            enabled.append(knowledge_id)
            config["enabled_knowledge_ids"] = enabled
            _save_config(config)
        click.echo(
            f"Enabled knowledge base "
            f"{knowledge.get('name') or knowledge_id} ({knowledge_id})"
        )

    @knowledge_group.command(name="disable")
    @click.argument("selector")
    def knowledge_disable(selector: str):
        """Stop attaching a persisted knowledge base to CLI chats."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")

        enabled = _enabled_knowledge_ids(config)
        knowledge_id: str
        name: str
        if selector in enabled:
            knowledge_id = selector
            name = selector
        else:
            client = _client(config)
            try:
                knowledge = _resolve_knowledge_selector(client, selector)
            except llm.ModelError as exc:
                raise click.ClickException(str(exc)) from exc
            knowledge_id = str(knowledge["id"])
            name = str(knowledge.get("name") or knowledge_id)

        config["enabled_knowledge_ids"] = [
            value for value in enabled if value != knowledge_id
        ]
        _save_config(config)
        click.echo(f"Disabled knowledge base {name} ({knowledge_id})")

    @knowledge_group.command(name="clear")
    def knowledge_clear():
        """Disable all CLI-persisted Open WebUI knowledge bases."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        config["enabled_knowledge_ids"] = []
        _save_config(config)
        click.echo("Disabled all CLI-persisted Open WebUI knowledge bases.")

    @knowledge_group.command(name="sync")
    @click.argument("selector")
    @click.argument(
        "source",
        required=False,
        type=click.Path(file_okay=False, path_type=Path),
    )
    def knowledge_sync(selector: str, source: Path | None):
        """Incrementally sync a local folder into an Open WebUI knowledge base."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            knowledge = _resolve_knowledge_selector(client, selector)
        except llm.ModelError as exc:
            raise click.ClickException(str(exc)) from exc

        knowledge_id = str(knowledge["id"])
        sources = _knowledge_sources(config)
        if source is None:
            stored = sources.get(knowledge_id)
            if not stored:
                raise click.ClickException(
                    "No local source folder is registered for this knowledge "
                    "base; provide PATH on this sync."
                )
            source = Path(stored)

        source = source.expanduser().resolve()
        status_bar = _OpenWebUIStatusBar()
        try:
            result = _sync_knowledge_folder(
                client,
                knowledge,
                source,
                on_status=status_bar.update,
                interactive_status=status_bar.enabled,
            )
        except (llm.ModelError, click.ClickException) as exc:
            status_bar.clear()
            raise click.ClickException(str(exc)) from exc
        finally:
            status_bar.clear()

        sources[knowledge_id] = str(source)
        config["knowledge_sources"] = sources
        _save_config(config)
        click.echo(
            f"Synced {knowledge.get('name') or knowledge_id}: "
            f"{result['added']} added, {result['modified']} modified, "
            f"{result['deleted']} deleted, {result['unmodified']} unchanged, "
            f"{result.get('reused', 0)} resumed."
        )

    @knowledge_group.command(name="rebuild")
    @click.argument("selector")
    @click.argument(
        "source",
        required=False,
        type=click.Path(file_okay=False, path_type=Path),
    )
    @click.option(
        "--yes",
        is_flag=True,
        help="Skip the destructive reset confirmation.",
    )
    def knowledge_rebuild(selector: str, source: Path | None, yes: bool):
        """Reset a knowledge base in place and repopulate it from its local folder."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)

        try:
            knowledge = _resolve_knowledge_selector(client, selector)
        except llm.ModelError as exc:
            raise click.ClickException(str(exc)) from exc

        if knowledge.get("write_access") is False:
            raise click.ClickException(
                f"Knowledge base {knowledge.get('name') or knowledge.get('id')} "
                "is read-only for this user."
            )

        knowledge_id = str(knowledge["id"])
        sources = _knowledge_sources(config)
        if source is None:
            stored = sources.get(knowledge_id)
            if not stored:
                raise click.ClickException(
                    "No local source folder is registered for this knowledge "
                    "base; provide PATH on this rebuild."
                )
            source = Path(stored)

        source = source.expanduser().resolve()
        # Validate and hash source before destructive reset so obvious local
        # problems cannot leave a previously healthy KB empty.
        manifest = _local_knowledge_manifest(source)
        if not manifest:
            raise click.ClickException(
                f"Knowledge source folder contains no visible files: {source}. "
                "Refusing to rebuild from an empty folder."
            )

        name = str(knowledge.get("name") or knowledge_id)
        if not yes:
            click.confirm(
                f"Reset and fully rebuild knowledge base {name!r} "
                f"({knowledge_id}) from {source}? This deletes the current KB "
                "index/file links before re-uploading.",
                abort=True,
            )

        status_bar = _OpenWebUIStatusBar()
        try:
            status_bar.update(f"knowledge rebuild · resetting {name}")
            _reset_knowledge_base(client, knowledge_id)
            status_bar.update(
                f"knowledge rebuild · repopulating {len(manifest)} files"
            )
            result = _sync_knowledge_folder(
                client,
                knowledge,
                source,
                on_status=status_bar.update,
                interactive_status=status_bar.enabled,
                detect_interrupted=False,
            )
        except (llm.ModelError, click.ClickException) as exc:
            status_bar.clear()
            raise click.ClickException(
                f"Knowledge rebuild failed after reset: {exc}"
            ) from exc
        finally:
            status_bar.clear()

        sources[knowledge_id] = str(source)
        config["knowledge_sources"] = sources
        _save_config(config)

        click.echo(
            f"Rebuilt {name} ({knowledge_id}) in place: "
            f"{result['uploaded']} uploaded, "
            f"{result['unmodified']} unchanged after reset."
        )

    @knowledge_group.command(name="register")
    @click.argument("name")
    @click.argument(
        "source",
        type=click.Path(file_okay=False, path_type=Path),
    )
    @click.option(
        "--description",
        default="Reusable knowledge synchronized by llm Open WebUI CLI.",
        show_default=True,
        help="Description used if the knowledge base must be created.",
    )
    def knowledge_register(name: str, source: Path, description: str):
        """Create/reuse, sync and enable a knowledge base in one command."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)

        try:
            available = _list_knowledge_bases(client)
        except llm.ModelError as exc:
            raise click.ClickException(str(exc)) from exc

        folded = name.casefold()
        matches = [
            item
            for item in available
            if str(item.get("name") or "").casefold() == folded
        ]
        if len(matches) > 1:
            detail = ", ".join(
                f"{item.get('name')} ({item.get('id')})" for item in matches
            )
            raise click.ClickException(
                f"Multiple knowledge bases have the exact name {name!r}: {detail}"
            )

        created = False
        if matches:
            knowledge = matches[0]
        else:
            try:
                knowledge = _create_knowledge_base(client, name, description)
            except llm.ModelError as exc:
                raise click.ClickException(str(exc)) from exc
            created = True

        if knowledge.get("write_access") is False:
            raise click.ClickException(
                f"Knowledge base {knowledge.get('name') or knowledge.get('id')} "
                "is read-only for this user."
            )

        source = source.expanduser().resolve()
        status_bar = _OpenWebUIStatusBar()
        try:
            result = _sync_knowledge_folder(
                client,
                knowledge,
                source,
                on_status=status_bar.update,
                interactive_status=status_bar.enabled,
            )
        except (llm.ModelError, click.ClickException) as exc:
            status_bar.clear()
            raise click.ClickException(str(exc)) from exc
        finally:
            status_bar.clear()

        knowledge_id = str(knowledge["id"])
        enabled = _enabled_knowledge_ids(config)
        if knowledge_id not in enabled:
            enabled.append(knowledge_id)
        sources = _knowledge_sources(config)
        sources[knowledge_id] = str(source)
        config["enabled_knowledge_ids"] = enabled
        config["knowledge_sources"] = sources
        _save_config(config)

        click.echo(
            f"{'Created' if created else 'Reused'} and registered "
            f"{knowledge.get('name') or knowledge_id} ({knowledge_id}); "
            f"{result['added']} added, {result['modified']} modified, "
            f"{result['deleted']} deleted, {result['unmodified']} unchanged, "
            f"{result.get('reused', 0)} resumed."
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
