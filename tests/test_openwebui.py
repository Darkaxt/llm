import io
import zipfile
import json

import pytest
import click
from types import SimpleNamespace

import llm
from llm.default_plugins import openwebui, openwebui_socket
from llm.parts import AttachmentPart, Message, TextPart


@pytest.fixture
def supported_openwebui_server(monkeypatch):
    """Mock the version endpoint; provider integration tests must stay offline."""
    monkeypatch.setattr(
        openwebui, "_server_config", lambda client: {"version": "0.11.3"}
    )


def test_prepare_openwebui_request_uses_full_chain():
    model = openwebui.OpenWebUIModel("glm")
    prompt = llm.Prompt(
        None,
        model,
        messages=[
            Message(role="system", parts=[TextPart(text="system")]),
            Message(role="user", parts=[TextPart(text="first")]),
            Message(role="assistant", parts=[TextPart(text="answer")]),
            Message(role="user", parts=[TextPart(text="second")]),
        ],
    )
    messages, files = openwebui._prepare_openwebui_request(
        prompt, SimpleNamespace(base_url="https://example.test")
    )
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    assert files == []


def test_prepare_openwebui_request_uploads_and_reuses_attachment(tmp_path, monkeypatch):
    path = tmp_path / "case.md"
    path.write_text("# Validation case\nCheck this file.")
    attachment = llm.Attachment(type="text/markdown", path=str(path))
    part = AttachmentPart(attachment=attachment)
    model = openwebui.OpenWebUIModel("glm")
    prompt = llm.Prompt(
        None,
        model,
        messages=[
            Message(
                role="user",
                parts=[TextPart(text="Validate this file"), part],
            )
        ],
    )
    uploaded = {
        "type": "file",
        "id": "file-1",
        "url": "file-1",
        "name": "case.md",
        "status": "uploaded",
        "content_type": "text/markdown",
        "context": "full",
        "file": {"id": "file-1"},
    }
    calls = []
    monkeypatch.setattr(
        openwebui,
        "_upload_attachment",
        lambda client, attachment, index, **kwargs: calls.append(index) or uploaded,
    )
    client = SimpleNamespace(base_url="https://example.test")

    messages, files = openwebui._prepare_openwebui_request(prompt, client)
    assert messages == [{"role": "user", "content": "Validate this file"}]
    assert files == [uploaded]
    assert calls == [1]

    # Provider metadata on the AttachmentPart prevents duplicate uploads on
    # subsequent turns / resumed message chains against the same server.
    messages, files = openwebui._prepare_openwebui_request(prompt, client)
    assert files == [uploaded]
    assert calls == [1]


def test_register_models_from_cache(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "models": [
                {"id": "glm-5.3", "name": "GLM 5.3"},
                {"id": "mcfly", "name": "MARTI - McFly"},
            ]
        },
    )
    registered = []

    def register(model, aliases=None):
        registered.append((model, aliases))

    openwebui.register_models(register)
    assert [item[0].model_id for item in registered] == [
        "openwebui/glm-5.3",
        "openwebui/mcfly",
    ]
    assert registered[0][1] == ["owui/glm-5.3"]


def test_execute_streams_text_and_reasoning(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {"url": "https://example.test", "token": "jwt", "models": []},
    )

    class FakeClient:
        def resolve_tools(self, model_id, extra_tool_ids=None, no_tools=False):
            assert model_id == "glm-5.3"
            assert no_tools is False
            assert extra_tool_ids == []
            return ["splunk"]

        def run_chat(self, **kwargs):
            kwargs["on_reasoning"]("thinking")
            kwargs["on_text"]("hello")
            kwargs["on_text"](" world")
            kwargs["on_tool"]("splunk_search")
            return SimpleNamespace(
                answer="hello world",
                reasoning="thinking",
                tool_calls=[{"name": "splunk_search"}],
            )

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda client, model_id: {
            "id": model_id,
            "name": "GLM",
            "info": {"meta": {"capabilities": {}}},
        },
    )

    model = openwebui.OpenWebUIModel("glm-5.3")
    prompt = llm.Prompt("test", model)
    response = SimpleNamespace(response_json=None)

    chunks = list(model.execute(prompt, True, response, None))
    assert chunks[0].type == "reasoning"
    assert chunks[0].chunk == "thinking"
    assert chunks[1:] == ["hello", " world"]
    assert response.response_json["remote_model"] == "glm-5.3"
    assert response.response_json["tool_activity"] == ["splunk_search"]
    assert response.response_json["raw_content"] == ""


def test_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(openwebui.llm, "user_dir", lambda: tmp_path)
    config = {
        "url": "https://example.test",
        "email": "user@example.test",
        "token": "jwt",
        "models": [{"id": "glm", "name": "GLM"}],
    }
    openwebui._save_config(config)
    assert openwebui._load_config() == config
    assert json.loads((tmp_path / "openwebui.json").read_text()) == config


def test_openwebui_model_accepts_general_attachment_types(tmp_path):
    path = tmp_path / "bundle.zip"
    path.write_bytes(b"PK\\x03\\x04test")
    model = openwebui.OpenWebUIModel("glm")
    # The provider defers file-policy enforcement to Open WebUI rather than
    # LLM's static attachment_types allowlist.
    model._validate_attachments(
        [llm.Attachment(type="application/zip", path=str(path))]
    )


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, statuses, file_payload=None):
        self.statuses = iter(statuses)
        self.file_payload = file_payload or {"data": {}}

    def get(self, url, **kwargs):
        if url.endswith("/process/status"):
            return _FakeResponse({"status": next(self.statuses)})
        return _FakeResponse(self.file_payload)


def test_wait_for_file_processing_completes(monkeypatch):
    monkeypatch.setattr(openwebui.time, "sleep", lambda _seconds: None)
    http = _FakeHttp(["pending", "processing", "completed"])
    client = SimpleNamespace(
        base_url="https://example.test",
        token="jwt",
    )
    openwebui._wait_for_file_processing(http, client, "file-1", "bundle.zip")


def test_wait_for_file_processing_surfaces_server_error(monkeypatch):
    monkeypatch.setattr(openwebui.time, "sleep", lambda _seconds: None)
    http = _FakeHttp(
        ["failed"],
        file_payload={"data": {"error": "unsupported archive"}},
    )
    client = SimpleNamespace(
        base_url="https://example.test",
        token="jwt",
    )
    try:
        openwebui._wait_for_file_processing(http, client, "file-1", "bundle.zip")
    except llm.ModelError as exc:
        assert "unsupported archive" in str(exc)
    else:
        raise AssertionError("expected file processing failure")



def test_attachment_context_auto_uses_full_for_small_text_and_rag_for_zip():
    files = [
        {
            "type": "file",
            "id": "md",
            "name": "SKILL.md",
            "content_type": "text/markdown",
            "size": 32_000,
        },
        {
            "type": "file",
            "id": "zip",
            "name": "kb.zip",
            "content_type": "application/zip",
            "size": 40_000,
        },
    ]
    openwebui._apply_attachment_context_policy(files, "auto")
    assert files[0]["context"] == "full"
    assert "context" not in files[1]


def test_attachment_context_rag_forces_chunked_retrieval():
    files = [
        {
            "type": "file",
            "id": "md",
            "name": "SPEC.md",
            "content_type": "text/markdown",
            "size": 12_000,
            "context": "full",
        }
    ]
    openwebui._apply_attachment_context_policy(files, "rag")
    assert "context" not in files[0]


def test_attachment_context_full_is_explicit_override():
    files = [
        {
            "type": "file",
            "id": "zip",
            "name": "kb.zip",
            "content_type": "application/zip",
            "size": 40_000,
        }
    ]
    openwebui._apply_attachment_context_policy(files, "full")
    assert files[0]["context"] == "full"



def test_openwebui_chat_timeout_default_and_override(monkeypatch):
    monkeypatch.delenv("LLM_OPENWEBUI_CHAT_TIMEOUT", raising=False)
    assert openwebui._chat_timeout() == 1200

    monkeypatch.setenv("LLM_OPENWEBUI_CHAT_TIMEOUT", "900")
    assert openwebui._chat_timeout() == 900



@pytest.mark.usefixtures("supported_openwebui_server")
def test_enabled_tool_ids_are_merged_into_runtime(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "models": [],
            "enabled_tool_ids": ["server:mcp:splunk"],
        },
    )

    seen = {}

    class FakeClient:
        base_url = "https://example.test"
        token = "jwt"
        timeout = 600

        def resolve_tools(self, model_id, extra_tool_ids=None, no_tools=False):
            seen["extra"] = extra_tool_ids
            return list(extra_tool_ids or [])

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda client, model_id: {
            "id": model_id,
            "name": "GLM",
            "info": {"meta": {"capabilities": {}}},
        },
    )

    async def fake_tool_runner(**kwargs):
        return {
            "answer": "ok",
            "reasoning": None,
            "tool_calls": [],
            "raw_content": "ok",
        }

    monkeypatch.setattr(
        openwebui,
        "run_chat_with_tools_with_files",
        fake_tool_runner,
    )

    model = openwebui.OpenWebUIModel("glm-5.3")
    prompt = llm.Prompt("test", model)
    response = SimpleNamespace(response_json=None)

    # No attachments means the SDK run_chat path would normally be used, so
    # inject a harmless attachment file list via the preparation helper to
    # exercise our attachment-aware socket path deterministically.
    monkeypatch.setattr(
        openwebui,
        "_prepare_openwebui_request",
        lambda prompt, client, **kwargs: (
            [{"role": "user", "content": "test"}],
            [{"id": "file-1", "type": "file"}],
        ),
    )
    calls = []

    async def capture_runner(**kwargs):
        calls.append(kwargs)
        return {
            "answer": "ok",
            "reasoning": None,
            "tool_calls": [],
            "raw_content": "ok",
        }

    monkeypatch.setattr(
        openwebui,
        "run_chat_with_tools_with_files",
        capture_runner,
    )

    assert list(model.execute(prompt, True, response, None)) == []
    assert len(calls) == 1
    assert calls[0]["tool_ids"] == ["server:mcp:splunk"]
    assert "extra" not in seen


def test_resolve_tool_selector_matches_mcp_name():
    tools = [
        SimpleNamespace(id="server:mcp:splunk-main", name="Splunk MCP"),
        SimpleNamespace(id="local-tool", name="Local Tool"),
    ]
    client = SimpleNamespace(list_tools=lambda: tools)

    tool = openwebui._resolve_tool_selector(client, "Splunk MCP")
    assert tool.id == "server:mcp:splunk-main"
    assert openwebui._tool_kind(tool.id) == "mcp"



@pytest.mark.usefixtures("supported_openwebui_server")
def test_tool_chat_without_attachments_uses_compat_runner(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "models": [],
            "enabled_tool_ids": ["server:mcp:splunk-mcp"],
        },
    )

    class FakeClient:
        base_url = "https://example.test"
        token = "jwt"
        timeout = 600

        def resolve_tools(self, model_id, extra_tool_ids=None, no_tools=False):
            return list(extra_tool_ids or [])

        def run_chat(self, **kwargs):
            raise AssertionError("SDK run_chat must not be used for tool-enabled chats")

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda client, model_id: {
            "id": model_id,
            "name": "GLM",
            "info": {"meta": {"capabilities": {}}},
        },
    )
    monkeypatch.setattr(
        openwebui,
        "_prepare_openwebui_request",
        lambda prompt, client, **kwargs: ([{"role": "user", "content": "test"}], []),
    )

    calls = []

    async def fake_runner(**kwargs):
        calls.append(kwargs)
        return {
            "answer": "ok",
            "reasoning": None,
            "tool_calls": [],
            "raw_content": "ok",
        }

    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", fake_runner)

    model = openwebui.OpenWebUIModel("glm-5.3")
    prompt = llm.Prompt("test", model)
    response = SimpleNamespace(response_json=None)

    assert list(model.execute(prompt, True, response, None)) == []
    assert len(calls) == 1
    assert calls[0]["tool_ids"] == ["server:mcp:splunk-mcp"]
    assert calls[0]["files"] == []
    assert calls[0]["model_item"]["info"]["meta"]["capabilities"] == {}



def test_current_openwebui_structured_output_parsing():
    output = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Need Splunk."}],
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "splunk-mcp_search",
            "arguments": "{\"search\": \"index=_internal | head 1\"}",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": [{"type": "output_text", "text": "{\"result\": 1}"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The search succeeded."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "output_text", "text": "Summarize the returned event."}],
        },
    ]

    assert (
        openwebui_socket._structured_output_text(output)
        == "The search succeeded."
    )
    assert openwebui_socket._structured_reasoning(output) == ["Need Splunk."]

    events = openwebui_socket._structured_tool_events(output)
    assert events == [
        {
            "name": "splunk-mcp_search",
            "call_id": "call-1",
            "arguments": "{\"search\": \"index=_internal | head 1\"}",
            "result": "{\"result\": 1}",
            "done": True,
        }
    ]



def test_file_progress_can_be_routed_to_status_callback(monkeypatch):
    seen = []
    monkeypatch.setattr(openwebui.time, "monotonic", lambda: 65.0)

    openwebui._render_file_progress(
        filename="bundle.zip",
        status="processing",
        started=5.0,
        tick=3,
        ordinal=3,
        total=3,
        on_status=seen.append,
    )

    assert seen == [
        "attachments 3/3 · bundle.zip · processing · 01:00"
    ]


def test_attachment_context_can_be_routed_to_status_callback():
    files = [
        {
            "type": "file",
            "id": "skill",
            "name": "SKILL.md",
            "size": 1024,
            "content_type": "text/markdown",
        },
        {
            "type": "file",
            "id": "kb",
            "name": "kb.zip",
            "size": 1024 * 1024,
            "content_type": "application/zip",
        },
    ]
    seen = []

    openwebui._apply_attachment_context_policy(
        files,
        "auto",
        on_status=seen.append,
    )

    assert files[0]["context"] == "full"
    assert "context" not in files[1]
    assert seen == [
        "attachment context · SKILL.md=full · kb.zip=rag"
    ]



def test_zip_sidecar_extraction_prefers_machine_readable_bundle_files():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "ec-tide-splunk-investigation-kb/rule-index.json",
            '{"67e794d5-73b2-45e5-b570-ceb5e0bba352":"rule.yaml"}',
        )
        archive.writestr(
            "ec-tide-splunk-investigation-kb/splunk-rules.jsonl",
            '{"uuid":"67e794d5-73b2-45e5-b570-ceb5e0bba352"}\n',
        )
        archive.writestr(
            "ec-tide-splunk-investigation-kb/macros.json",
            '{"macro":"value"}',
        )
        archive.writestr(
            "ec-tide-splunk-investigation-kb/rules/unrelated.yaml",
            "name: unrelated",
        )

    attachment = llm.Attachment(
        type="application/zip",
        content=payload.getvalue(),
    )
    sidecars = openwebui._zip_sidecar_attachments(
        attachment,
        filename="ec-tide-splunk-investigation-kb.zip",
    )

    assert [name for name, _ in sidecars] == [
        "rule-index.json",
        "splunk-rules.jsonl",
        "macros.json",
    ]
    assert b"67e794d5-73b2-45e5-b570-ceb5e0bba352" in sidecars[0][1].content
    assert sidecars[1][1].type == "application/x-ndjson"
    assert sidecars[2][1].type == "application/json"


def test_zip_without_expected_sidecars_is_left_unchanged():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("notes/readme.txt", "hello")

    attachment = llm.Attachment(
        type="application/zip",
        content=payload.getvalue(),
    )

    assert (
        openwebui._zip_sidecar_attachments(
            attachment,
            filename="other.zip",
        )
        == []
    )



def test_sidecar_member_selection_prefers_one_coherent_root():
    selected = openwebui._select_sidecar_members(
        [
            "bundle/rule-index.json",
            "bundle/splunk-rules.jsonl",
            "bundle/macros.json",
            "docs/rule-index.json",
        ]
    )

    assert selected == {
        "rule-index.json": "bundle/rule-index.json",
        "splunk-rules.jsonl": "bundle/splunk-rules.jsonl",
        "macros.json": "bundle/macros.json",
    }


def test_sidecar_member_selection_rejects_two_complete_roots():
    with pytest.raises(llm.ModelError, match="multiple complete TIDE sidecar sets"):
        openwebui._select_sidecar_members(
            [
                "bundle-a/rule-index.json",
                "bundle-a/splunk-rules.jsonl",
                "bundle-a/macros.json",
                "bundle-b/rule-index.json",
                "bundle-b/splunk-rules.jsonl",
                "bundle-b/macros.json",
            ]
        )


def test_sidecar_member_selection_rejects_ambiguous_duplicate_basename():
    with pytest.raises(llm.ModelError, match="ambiguous TIDE sidecar filenames"):
        openwebui._select_sidecar_members(
            [
                "a/rule-index.json",
                "b/rule-index.json",
                "splunk-rules.jsonl",
                "macros.json",
            ]
        )


def test_7z_sidecar_extraction():
    import py7zr

    payload = io.BytesIO()
    with py7zr.SevenZipFile(payload, "w") as archive:
        archive.writestr(
            '{"67e794d5-73b2-45e5-b570-ceb5e0bba352":"rule.yaml"}',
            "ec-tide-splunk-investigation-kb/rule-index.json",
        )
        archive.writestr(
            '{"uuid":"67e794d5-73b2-45e5-b570-ceb5e0bba352"}\n',
            "ec-tide-splunk-investigation-kb/splunk-rules.jsonl",
        )
        archive.writestr(
            '{"macro":"value"}',
            "ec-tide-splunk-investigation-kb/macros.json",
        )

    attachment = llm.Attachment(
        type="application/x-7z-compressed",
        content=payload.getvalue(),
    )
    sidecars = openwebui._archive_sidecar_attachments(
        attachment,
        filename="ec-tide-splunk-investigation-kb.7z",
    )

    assert [name for name, _ in sidecars] == [
        "rule-index.json",
        "splunk-rules.jsonl",
        "macros.json",
    ]
    assert b"67e794d5-73b2-45e5-b570-ceb5e0bba352" in sidecars[0][1].content



def test_local_knowledge_manifest_preserves_relative_paths(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "rule.yaml").write_text("a", encoding="utf-8")
    (tmp_path / "b" / "rule.yaml").write_text("b", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "skip.txt").write_text("skip", encoding="utf-8")

    manifest = openwebui._local_knowledge_manifest(tmp_path)

    assert [(item["path"], item["filename"]) for item in manifest] == [
        ("a", "rule.yaml"),
        ("b", "rule.yaml"),
    ]
    assert manifest[0]["checksum"] != manifest[1]["checksum"]


def test_enabled_knowledge_items_match_browser_picker_shape(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_list_knowledge_bases",
        lambda client: [
            {
                "id": "kb-1",
                "name": "TIDE Splunk Investigation",
                "description": "Reusable TIDE knowledge",
                "user_id": "user-1",
                "data": {"file_ids": ["file-1"]},
                "meta": {"source": "local"},
                "access_grants": [{"permission": "read"}],
                "write_access": True,
            }
        ],
    )
    config = {"enabled_knowledge_ids": ["kb-1"]}
    client = SimpleNamespace()

    items = openwebui._enabled_knowledge_items(config, client)

    assert items == [
        {
            "type": "collection",
            "id": "kb-1",
            "name": "TIDE Splunk Investigation",
            "description": "Reusable TIDE knowledge",
            "user_id": "user-1",
            "data": {"file_ids": ["file-1"]},
            "meta": {"source": "local"},
            "access_grants": [{"permission": "read"}],
            "write_access": True,
        }
    ]


@pytest.mark.usefixtures("supported_openwebui_server")
@pytest.mark.parametrize(
    "transport, sessionless, function_calling",
    [
        ("background_legacy", False, "legacy"),
        ("background_native", False, "native"),
    ],
)
def test_enabled_knowledge_is_prefetched_without_forced_rag(
    monkeypatch, transport, sessionless, function_calling
):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "models": [],
            "enabled_tool_ids": ["server:mcp:splunk-mcp"],
            "enabled_knowledge_ids": ["kb-1"],
        },
    )

    calls = []

    class FakeClient:
        base_url = "https://example.test"
        token = "jwt"
        timeout = 600

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda client, model_id: {
            "id": model_id,
            "name": "GLM",
            "info": {"meta": {"capabilities": {"builtin_tools": False}}},
        },
    )
    monkeypatch.setattr(
        openwebui,
        "_prepare_openwebui_request",
        lambda prompt, client, **kwargs: (
            [{"role": "user", "content": "investigate"}],
            [],
        ),
    )
    monkeypatch.setattr(
        openwebui,
        "_enabled_knowledge_items",
        lambda config, client: [
            {
                "type": "collection",
                "id": "kb-1",
                "name": "TIDE Splunk Investigation",
                "description": "Reusable TIDE knowledge",
                "write_access": True,
            }
        ],
    )
    monkeypatch.setattr(
        openwebui,
        "_resolve_knowledge_context",
        lambda client, knowledge_items, messages, on_status=None: (
            [
                {
                    "role": "system",
                    "content": "<persistent_knowledge>resolved TIDE</persistent_knowledge>",
                },
                *messages,
            ],
            {
                "knowledge_bases": [
                    {"id": "kb-1", "name": "TIDE Splunk Investigation"}
                ],
                "context_chars": 55,
            },
        ),
    )

    async def fake_runner(**kwargs):
        calls.append(kwargs)
        return {
            "answer": "ok",
            "reasoning": None,
            "tool_calls": [],
            "raw_content": "ok",
        }

    monkeypatch.setattr(
        openwebui,
        "run_chat_with_tools_with_files",
        fake_runner,
    )

    model = openwebui.OpenWebUIModel("glm-5.3")
    prompt = llm.Prompt(
        "investigate",
        model,
        options=model.Options(openwebui_mcp_transport=transport),
    )
    response = SimpleNamespace(response_json=None)

    assert list(model.execute(prompt, True, response, None)) == []
    assert len(calls) == 1
    assert calls[0]["files"] == []
    assert calls[0]["tool_ids"] == ["server:mcp:splunk-mcp"]
    assert calls[0]["params"].get("function_calling") == function_calling
    assert calls[0]["sessionless_server_tools"] is sessionless
    assert calls[0]["messages"][0]["role"] == "system"
    assert "resolved TIDE" in calls[0]["messages"][0]["content"]
    assert calls[0]["messages"][1] == {
        "role": "user",
        "content": "investigate",
    }


def test_resolve_knowledge_context_extracts_exact_tide_evidence(monkeypatch):
    uuid_value = "67e794d5-73b2-45e5-b570-ceb5e0bba352"
    contents = {
        "skill": "# SKILL\nFollow the workflow.",
        "spec": "# SPEC\nEvidence requirements.",
        "index": json.dumps(
            {
                uuid_value: {
                    "name": "CSOC integration for AWS GUARDDUTY",
                    "rule_file": "guardduty.yaml",
                },
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": {
                    "name": "unrelated"
                },
            }
        ),
        "rules": "\n".join(
            [
                json.dumps(
                    {
                        "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                        "search": "index=other",
                    }
                ),
                json.dumps(
                    {
                        "uuid": uuid_value,
                        "name": "CSOC integration for AWS GUARDDUTY",
                        "search": "`aws_cloudtrail` eventName=CreateKeyPair",
                    }
                ),
            ]
        ),
        "macros": json.dumps(
            {
                "aws_cloudtrail": {
                    "definition": "index=aws sourcetype=aws:cloudtrail"
                },
                "unrelated_macro": {"definition": "index=other"},
            }
        ),
    }
    file_map = {
        "SKILL.md": {"id": "skill", "filename": "SKILL.md"},
        "SPEC.md": {"id": "spec", "filename": "SPEC.md"},
        "rule-index.json": {"id": "index", "filename": "rule-index.json"},
        "splunk-rules.jsonl": {
            "id": "rules",
            "filename": "splunk-rules.jsonl",
        },
        "macros.json": {"id": "macros", "filename": "macros.json"},
    }

    monkeypatch.setattr(
        openwebui,
        "_knowledge_exact_file",
        lambda client, knowledge_id, filename: file_map.get(filename),
    )
    monkeypatch.setattr(
        openwebui,
        "_knowledge_file_text",
        lambda client, file_id: contents[file_id],
    )

    messages, meta = openwebui._resolve_knowledge_context(
        SimpleNamespace(),
        [
            {
                "id": "kb-1",
                "name": "TIDE Splunk Investigation",
            }
        ],
        [
            {
                "role": "user",
                "content": f"Investigate exact MDR UUID {uuid_value}",
            }
        ],
    )

    assert messages[0]["role"] == "system"
    context = messages[0]["content"]
    assert "# SKILL" in context
    assert "# SPEC" in context
    assert uuid_value in context
    assert "CSOC integration for AWS GUARDDUTY" in context
    assert "aws_cloudtrail" in context
    assert "index=aws sourcetype=aws:cloudtrail" in context
    assert "unrelated_macro" not in context
    assert "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" not in context
    assert messages[1]["role"] == "user"
    assert meta["uuids"] == [uuid_value]
    assert meta["context_chars"] == len(context)


def test_resolve_knowledge_context_appends_existing_system_message(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_knowledge_exact_file",
        lambda client, knowledge_id, filename: (
            {"id": "skill", "filename": "SKILL.md"}
            if filename == "SKILL.md"
            else None
        ),
    )
    monkeypatch.setattr(
        openwebui,
        "_knowledge_file_text",
        lambda client, file_id: "# Skill",
    )

    original = [
        {"role": "system", "content": "existing system"},
        {"role": "user", "content": "test"},
    ]
    scoped, _ = openwebui._resolve_knowledge_context(
        SimpleNamespace(),
        [{"id": "kb-1", "name": "TIDE"}],
        original,
    )

    assert scoped[0]["content"].startswith("existing system\n\n")
    assert "# Skill" in scoped[0]["content"]
    assert original[0]["content"] == "existing system"


def test_sync_knowledge_folder_uses_native_diff_and_relative_paths(
    tmp_path, monkeypatch
):
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "one.yaml").write_text("one", encoding="utf-8")
    (tmp_path / "macros.json").write_text("{}", encoding="utf-8")

    requests = []
    uploads = []

    def fake_json(client, method, path, **kwargs):
        requests.append((method, path, kwargs))
        if path.endswith("/sync/diff"):
            return {
                "added": [
                    {"filename": "one.yaml", "path": "rules"},
                    {"filename": "macros.json", "path": ""},
                ],
                "modified": [],
                "deleted": [],
                "mkdir": ["rules"],
                "rmdir": [],
                "unmodified_count": 0,
                "directory_map": {},
            }
        if path.endswith("/dirs/create"):
            return {"id": "dir-rules"}
        raise AssertionError(path)

    monkeypatch.setattr(openwebui, "_owui_http_json", fake_json)
    monkeypatch.setattr(
        openwebui,
        "_upload_knowledge_sync_file",
        lambda client, **kwargs: uploads.append(kwargs) or {"id": "file"},
    )
    monkeypatch.setattr(
        openwebui,
        "_knowledge_file_candidates",
        lambda client, knowledge_id: {},
    )

    client = SimpleNamespace(
        base_url="https://example.test",
        token="jwt",
        timeout=600,
    )
    knowledge = {"id": "kb-1", "name": "TIDE"}

    result = openwebui._sync_knowledge_folder(
        client,
        knowledge,
        tmp_path,
    )

    diff_body = requests[0][2]["json_body"]
    assert {
        (item["path"], item["filename"])
        for item in diff_body["manifest"]
    } == {
        ("rules", "one.yaml"),
        ("", "macros.json"),
    }
    assert len(uploads) == 2
    assert {
        (item["entry"]["path"], item["entry"]["filename"])
        for item in uploads
    } == {
        ("rules", "one.yaml"),
        ("", "macros.json"),
    }
    assert result == {
        "added": 2,
        "modified": 0,
        "deleted": 0,
        "unmodified": 0,
        "uploaded": 2,
        "reused": 0,
    }


def test_local_knowledge_manifest_expands_zip_as_virtual_tree(tmp_path):
    (tmp_path / "SKILL.md").write_text("# skill", encoding="utf-8")
    (tmp_path / "SPEC.md").write_text("# spec", encoding="utf-8")

    archive_path = tmp_path / "ec-tide-splunk-investigation-kb.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "ec-tide-splunk-investigation-kb/rule-index.json",
            '{"rule":"value"}',
        )
        archive.writestr(
            "ec-tide-splunk-investigation-kb/rules/aws/rule.yaml",
            "name: aws rule",
        )

    manifest = openwebui._local_knowledge_manifest(tmp_path)
    virtual_paths = {
        (item["path"], item["filename"])
        for item in manifest
    }

    assert ("", "SKILL.md") in virtual_paths
    assert ("", "SPEC.md") in virtual_paths
    assert (
        "ec-tide-splunk-investigation-kb",
        "rule-index.json",
    ) in virtual_paths
    assert (
        "ec-tide-splunk-investigation-kb/rules/aws",
        "rule.yaml",
    ) in virtual_paths
    assert ("", "ec-tide-splunk-investigation-kb.zip") not in virtual_paths


def test_local_knowledge_manifest_rejects_archive_virtual_path_collision(tmp_path):
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "same.txt").write_text("local", encoding="utf-8")
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bundle/same.txt", "archive")

    with pytest.raises(click.ClickException, match="duplicate virtual path"):
        openwebui._local_knowledge_manifest(tmp_path)


def test_local_knowledge_manifest_expands_7z_as_virtual_tree(tmp_path):
    import py7zr

    archive_path = tmp_path / "bundle.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.writestr(
            '{"rule":"value"}',
            "bundle/rule-index.json",
        )

    manifest = openwebui._local_knowledge_manifest(tmp_path)
    virtual_paths = {
        (item["path"], item["filename"])
        for item in manifest
    }

    assert ("bundle", "rule-index.json") in virtual_paths
    assert ("", "bundle.7z") not in virtual_paths

def test_interrupted_knowledge_sync_fails_before_mutation(tmp_path, monkeypatch):
    (tmp_path / "rule-index.json").write_text("{}", encoding="utf-8")
    checksum = openwebui.hashlib.sha256(b"{}").hexdigest()
    requests = []

    def fake_json(client, method, path, **kwargs):
        requests.append((method, path, kwargs))
        if path.endswith("/sync/diff"):
            return {
                "added": [{"filename": "rule-index.json", "path": ""}],
                "modified": [],
                "deleted": [{"file_id": "stale", "filename": "old.txt"}],
                "mkdir": [],
                "rmdir": ["old-dir"],
                "unmodified_count": 0,
                "directory_map": {},
            }
        raise AssertionError(
            f"Sync mutated server before interrupted-state failure: {path}"
        )

    monkeypatch.setattr(openwebui, "_owui_http_json", fake_json)
    monkeypatch.setattr(
        openwebui,
        "_knowledge_file_candidates",
        lambda client, knowledge_id: {
            checksum: [
                {
                    "id": "prior-file",
                    "filename": "rule-index.json",
                    "meta": {
                        "name": "rule-index.json",
                        "file_hash": checksum,
                        "data": {"knowledge_id": "kb-1"},
                    },
                }
            ]
        },
    )

    client = SimpleNamespace(timeout=600)
    knowledge = {"id": "kb-1", "name": "TIDE Splunk Investigation"}

    with pytest.raises(
        click.ClickException,
        match="knowledge rebuild",
    ):
        openwebui._sync_knowledge_folder(
            client,
            knowledge,
            tmp_path,
        )

    assert len(requests) == 1
    assert requests[0][1].endswith("/sync/diff")


def test_reset_knowledge_base_preserves_id_and_resets_directories(monkeypatch):
    calls = []

    def fake_json(client, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"id": "kb-1", "name": "TIDE Splunk Investigation"}

    monkeypatch.setattr(openwebui, "_owui_http_json", fake_json)

    result = openwebui._reset_knowledge_base(
        SimpleNamespace(timeout=600),
        "kb-1",
    )

    assert result["id"] == "kb-1"
    assert calls == [
        (
            "POST",
            "/api/v1/knowledge/kb-1/reset",
            {
                "params": {"include_directories": "true"},
                "timeout": 600.0,
            },
        )
    ]


def test_clean_rebuild_sync_bypasses_orphan_detection(tmp_path, monkeypatch):
    (tmp_path / "rule-index.json").write_text("{}", encoding="utf-8")
    requests = []
    uploads = []

    def fake_json(client, method, path, **kwargs):
        requests.append((method, path, kwargs))
        if path.endswith("/sync/diff"):
            return {
                "added": [{"filename": "rule-index.json", "path": ""}],
                "modified": [],
                "deleted": [],
                "mkdir": [],
                "rmdir": [],
                "unmodified_count": 0,
                "directory_map": {},
            }
        raise AssertionError(path)

    monkeypatch.setattr(openwebui, "_owui_http_json", fake_json)
    monkeypatch.setattr(
        openwebui,
        "_knowledge_file_candidates",
        lambda client, knowledge_id: (_ for _ in ()).throw(
            AssertionError("orphan inventory must be bypassed during rebuild")
        ),
    )
    monkeypatch.setattr(
        openwebui,
        "_upload_knowledge_sync_file",
        lambda client, **kwargs: uploads.append(kwargs) or {"id": "new-file"},
    )

    result = openwebui._sync_knowledge_folder(
        SimpleNamespace(timeout=600),
        {"id": "kb-1", "name": "TIDE"},
        tmp_path,
        detect_interrupted=False,
    )

    assert result["uploaded"] == 1
    assert len(uploads) == 1



def test_browser_chat_body_matches_current_openwebui_contract(monkeypatch):
    monkeypatch.setattr(openwebui_socket.time, "time", lambda: 1234.0)
    model_item = {
        "id": "deepseek-v41-flash",
        "name": "DeepSeek V4.1 Flash",
        "info": {"meta": {"capabilities": {"file_upload": True}}},
    }
    collection = {
        "type": "collection",
        "id": "kb-1",
        "name": "TIDE Splunk Investigation",
        "write_access": True,
    }

    body = openwebui_socket._build_browser_chat_body(
        model="deepseek-v41-flash",
        model_item=model_item,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "investigate"},
        ],
        tool_ids=["server:mcp:splunk-mcp"],
        files=[collection],
        params={"temperature": 0.2},
        chat_id="temporary:socket-1",
        session_id="socket-1",
        message_id="assistant-1",
        user_message_id="user-1",
    )

    assert body["model_item"] == model_item
    assert body["files"] == [collection]
    assert body["tool_ids"] == ["server:mcp:splunk-mcp"]
    assert body["params"] == {"temperature": 0.2}
    assert body["parent_id"] is None
    assert body["chat_variables"] == {}
    assert body["tool_servers"] == []
    assert body["background_tasks"] == {}
    assert body["user_message"] == {
        "id": "user-1",
        "parentId": None,
        "childrenIds": ["assistant-1"],
        "role": "user",
        "content": "investigate",
        "timestamp": 1234,
    }



def test_browser_chat_body_can_omit_session_id_for_server_tools(monkeypatch):
    monkeypatch.setattr(openwebui_socket.time, "time", lambda: 1234.0)

    body = openwebui_socket._build_browser_chat_body(
        model="deepseek-v41-flash",
        model_item={"id": "deepseek-v41-flash"},
        messages=[{"role": "user", "content": "investigate"}],
        tool_ids=["server:mcp:splunk-mcp"],
        files=[],
        params={},
        chat_id="temporary:listener-socket",
        session_id=None,
        message_id="assistant-1",
        user_message_id="user-1",
    )

    assert "session_id" not in body
    assert body["chat_id"] == "temporary:listener-socket"
    assert body["tool_ids"] == ["server:mcp:splunk-mcp"]
    assert body["params"] == {}

@pytest.mark.usefixtures("supported_openwebui_server")
def test_cli_tool_selection_does_not_merge_model_default_tools(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "models": [],
            "enabled_tool_ids": ["server:mcp:splunk-mcp"],
        },
    )

    class FakeClient:
        base_url = "https://example.test"
        token = "jwt"
        timeout = 600

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda client, model_id: {
            "id": model_id,
            "info": {
                "meta": {
                    "toolIds": [
                        "server:mcp:splunk-mcp",
                        "server:mcp:misp-prod",
                    ],
                    "capabilities": {"builtin_tools": True},
                }
            },
        },
    )
    monkeypatch.setattr(
        openwebui,
        "_prepare_openwebui_request",
        lambda prompt, client, **kwargs: (
            [{"role": "user", "content": "test"}],
            [],
        ),
    )
    monkeypatch.setattr(
        openwebui,
        "_enabled_knowledge_items",
        lambda config, client: [],
    )

    calls = []

    async def fake_runner(**kwargs):
        calls.append(kwargs)
        return {
            "answer": "ok",
            "reasoning": None,
            "tool_calls": [],
            "raw_content": "ok",
        }

    monkeypatch.setattr(
        openwebui,
        "run_chat_with_tools_with_files",
        fake_runner,
    )

    model = openwebui.OpenWebUIModel("deepseek-v41-flash")
    response = SimpleNamespace(response_json=None)
    assert list(
        model.execute(llm.Prompt("test", model), True, response, None)
    ) == []

    assert calls[0]["tool_ids"] == ["server:mcp:splunk-mcp"]
    assert "server:mcp:misp-prod" not in calls[0]["tool_ids"]


def test_server_mcp_guard_blocks_versions_before_0_9_3(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_server_config",
        lambda client: {"version": "0.9.2"},
    )
    monkeypatch.delenv(
        "LLM_OPENWEBUI_ALLOW_UNSAFE_SERVER_MCP",
        raising=False,
    )

    with pytest.raises(
        llm.ModelError,
        match="versions before 0.9.3",
    ):
        openwebui._guard_server_mcp_version(
            SimpleNamespace(),
            ["server:mcp:splunk-mcp"],
        )


def test_server_mcp_guard_allows_0_9_3_and_newer(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_server_config",
        lambda client: {"version": "0.9.3"},
    )

    assert openwebui._guard_server_mcp_version(
        SimpleNamespace(),
        ["server:mcp:splunk-mcp"],
    ) == ("0.9.3", True)


def test_server_mcp_guard_ignores_non_mcp_tools(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_server_config",
        lambda client: (_ for _ in ()).throw(
            AssertionError("/api/config should not be called")
        ),
    )

    assert openwebui._guard_server_mcp_version(
        SimpleNamespace(),
        ["local-tool"],
    ) == (None, None)


def test_server_mcp_guard_can_be_explicitly_overridden(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_server_config",
        lambda client: {"version": "0.9.1"},
    )
    monkeypatch.setenv(
        "LLM_OPENWEBUI_ALLOW_UNSAFE_SERVER_MCP",
        "1",
    )

    assert openwebui._guard_server_mcp_version(
        SimpleNamespace(),
        ["server:mcp:splunk-mcp"],
    ) == ("0.9.1", False)



def test_sessionless_http_drop_can_recover_completed_socket_event():
    import asyncio

    async def scenario():
        done = asyncio.Event()
        state = {"error": None}
        statuses = []

        async def complete_soon():
            await asyncio.sleep(0)
            done.set()

        task = asyncio.create_task(complete_soon())
        await openwebui_socket._recover_sessionless_completion(
            done,
            state,
            OSError(64, "network connection lost"),
            statuses.append,
            grace_seconds=0.1,
        )
        await task
        assert state["error"] is None
        assert any("HTTP connection lost" in value for value in statuses)

    asyncio.run(scenario())


def test_sessionless_http_drop_avoids_automatic_replay():
    import asyncio

    async def scenario():
        done = asyncio.Event()
        state = {"error": None}
        await openwebui_socket._recover_sessionless_completion(
            done,
            state,
            OSError(64, "network connection lost"),
            lambda status: None,
            grace_seconds=0.01,
        )
        assert "may still be running" in state["error"]
        assert "!retry" in state["error"]

    asyncio.run(scenario())


def test_sessionless_recovery_grace_validation(monkeypatch):
    monkeypatch.setenv("LLM_OPENWEBUI_RECOVERY_GRACE", "nan")
    with pytest.raises(Exception, match="must be non-negative"):
        openwebui_socket._sessionless_recovery_grace()
    monkeypatch.setenv("LLM_OPENWEBUI_RECOVERY_GRACE", "15")
    assert openwebui_socket._sessionless_recovery_grace() == 15.0



def test_meaningful_progress_timeout_not_reset_by_socket_keepalives():
    state = {
        "phase": "initial model",
        "last_event_at": 400.0,
        "last_progress_at": 0.0,
    }
    reason = openwebui_socket._stalled_request_error(
        state, 400.0, event_timeout=600.0, progress_timeout=300.0
    )
    assert "no model/tool progress for 400s" in reason
    assert "initial model" in reason
    state["last_progress_at"] = 350.0
    assert openwebui_socket._stalled_request_error(
        state, 400.0, event_timeout=600.0, progress_timeout=300.0
    ) is None


def test_meaningful_progress_timeout_config(monkeypatch):
    monkeypatch.delenv("LLM_OPENWEBUI_PROGRESS_TIMEOUT", raising=False)
    assert openwebui_socket._meaningful_progress_timeout() == 1200.0
    monkeypatch.setenv("LLM_OPENWEBUI_PROGRESS_TIMEOUT", "120")
    assert openwebui_socket._meaningful_progress_timeout() == 120.0
    for invalid in ("0", "-1", "nan", "inf", "oops"):
        monkeypatch.setenv("LLM_OPENWEBUI_PROGRESS_TIMEOUT", invalid)
        with pytest.raises(Exception, match="LLM_OPENWEBUI_PROGRESS_TIMEOUT"):
            openwebui_socket._meaningful_progress_timeout()


def test_cancel_remote_background_chat_tasks_uses_user_scoped_endpoint():
    import asyncio

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    async def scenario():
        session = Session()
        statuses = []
        result = await openwebui_socket._stop_remote_chat_tasks(
            session,
            base_url="https://example.test/",
            token="test-jwt",
            chat_id="temporary:session-1",
            on_status=statuses.append,
        )
        assert result is True
        assert session.calls == [
            (
                "https://example.test/api/tasks/chat/temporary%3Asession-1/stop",
                {
                    "headers": {"Authorization": "Bearer test-jwt"},
                    "timeout": 15,
                },
            )
        ]
        assert "cancellation requested" in statuses[0]

    asyncio.run(scenario())



@pytest.fixture
def tide_v3_knowledge_server(monkeypatch):
    """Simulate Open WebUI 0.11.3's directory-scoped Knowledge files API."""
    uuid = "67e794d5-73b2-45e5-b570-ceb5e0bba352"
    dirs = {
        "": [{"id": "bundle", "name": "ec-tide-splunk-investigation-kb"}],
        "bundle": [
            {"id": "rules", "name": "rules"},
            {"id": "macros", "name": "macros"},
        ],
        "rules": [{"id": "rule", "name": uuid}],
        "rule": [],
        "macros": [{"id": "by-stanza", "name": "by-stanza"}],
        "by-stanza": [],
    }
    contents = {
        "skill": (
            "# Skill 0.3.0\nBundle has macros/by-stanza/ microfiles.\n"
            "## Default route\nRule search.\n## Custom macros\n"
            "Read only macros/index.tsv.\n## Lookups\nNever read lookups.\n"
        ),
        "spec": "# Specification\nOnly read the selected TIDE microfiles.\n",
        "index": (
            "stanza\trole_hints\n"
            "aws_sso_log(1)\tsource\n"
            "aws_cloudtrail_log\tsource\n"
            "other_macro\tenrichment\n"
        ),
        "aws_sso_log": json.dumps({
            "stanza": "aws_sso_log(1)",
            "definition": "index=aws sourcetype=cloudtrail",
            "role_hints": ["source"],
        }),
        "aws_cloudtrail_log": json.dumps({
            "stanza": "aws_cloudtrail_log",
            "definition": "index=aws sourcetype=aws:cloudtrail",
            "role_hints": ["source"],
        }),
        "detection": '{"name": "CSOC integration for AWS GUARDDUTY"}',
        "query": "index=aws sourcetype=aws:asl:guardduty",
        "dependencies": json.dumps({
            "dependencies": [
                {
                    "resolution_policy": "resolve",
                    "file": "macros/by-stanza/aws_sso_log%281%29.json",
                },
                {
                    "resolution_policy": "skip",
                    "file": "macros/by-stanza/skipped_macro.json",
                },
            ]
        }),
    }
    files = {
        "macros": {"index.tsv": "index"},
        "by-stanza": {
            "aws_sso_log%281%29.json": "aws_sso_log",
            "aws_cloudtrail_log.json": "aws_cloudtrail_log",
        },
        "rule": {
            "detection.json": "detection",
            "query.spl": "query",
            "dependencies.json": "dependencies",
        },
    }
    requests = []

    def fake_json(client, method, endpoint, **kwargs):
        assert method == "GET"
        assert endpoint == "/api/v1/knowledge/kb-1/files"
        params = kwargs["params"]
        requests.append(params)
        parent = params["directory_id"]
        query = params.get("query")
        if query is None:
            return {"items": [], "directories": dirs.get(parent, []), "total": 0}
        file_id = files.get(parent, {}).get(query)
        return {
            "items": [{"id": file_id, "filename": query}] if file_id else [],
            "directories": dirs.get(parent, []),
            "total": int(file_id is not None),
        }

    monkeypatch.setattr(openwebui, "_owui_http_json", fake_json)
    monkeypatch.setattr(
        openwebui,
        "_knowledge_exact_file",
        lambda client, knowledge_id, filename: (
            {"id": filename.lower()[:-3], "filename": filename}
            if filename in {"SKILL.md", "SPEC.md"}
            else pytest.fail("v0.3 must not fetch obsolete aggregate files")
        ),
    )
    # The above IDs map SKILL.md -> skill and SPEC.md -> spec.
    monkeypatch.setattr(
        openwebui, "_knowledge_file_text", lambda client, file_id: contents[file_id]
    )
    return SimpleNamespace(
        client=SimpleNamespace(base_url="https://example.test", token="jwt", timeout=600),
        requests=requests,
        contents=contents,
        uuid=uuid,
    )


def test_tide_v3_macro_suffix_enumeration_reads_nested_index_and_definitions(
    tide_v3_knowledge_server,
):
    server = tide_v3_knowledge_server
    messages, meta = openwebui._resolve_knowledge_context(
        server.client,
        [{"id": "kb-1", "name": "TIDE Splunk Investigation"}],
        [{"role": "user", "content": (
            "Enumerate macros whose name ends in _log and classify their categories"
        )}],
    )
    context = messages[0]["content"]
    assert 'suffix="_log"' in context
    assert 'total_matches="2"' in context
    assert "aws_sso_log(1)" in context
    assert "aws_cloudtrail_log" in context
    assert '"role_hints": ["source"]' in context
    assert "other_macro" not in context
    assert "macros/by-stanza/aws_sso_log%281%29.json" in context
    assert any(item.get("selected_rows") == 2 for item in meta["files"])
    assert len(context) < 12000
    assert not any("rule-index.json" in str(req) for req in server.requests)


def test_tide_v3_uuid_fetches_exact_rule_and_resolve_only_macros(
    tide_v3_knowledge_server,
):
    server = tide_v3_knowledge_server
    messages, meta = openwebui._resolve_knowledge_context(
        server.client,
        [{"id": "kb-1", "name": "TIDE Splunk Investigation"}],
        [{"role": "user", "content": f"Investigate MDR_UUID={server.uuid}"}],
    )
    context = messages[0]["content"]
    for path in (
        f"rules/{server.uuid}/detection.json",
        f"rules/{server.uuid}/query.spl",
        f"rules/{server.uuid}/dependencies.json",
        "macros/by-stanza/aws_sso_log%281%29.json",
    ):
        assert path in context
    assert "index=aws sourcetype=aws:asl:guardduty" in context
    assert "skipped_macro.json" not in [item.get("path") for item in meta["files"]]
    assert "skipped_macro" not in context.split("<file path=")[-1]
    assert not any(req.get("query") == "skipped_macro.json" for req in server.requests)


def test_tide_v3_target_prefers_mdr_uuid_over_other_notable_ids():
    selected = "67e794d5-73b2-45e5-b570-ceb5e0bba352"
    unrelated = "1cf1610c-541c-499e-84ee-075dfaa2cdeb"
    messages = [{
        "role": "user",
        "content": f'MDR_UUID="{selected}", MDR_detection_model="{unrelated}"',
    }]
    assert openwebui._tide_target_rule_uuids(
        messages, {selected, unrelated}
    ) == [selected]


def test_tide_v3_exact_path_no_global_basename_collision(
    tide_v3_knowledge_server,
):
    server = tide_v3_knowledge_server
    cache = {}
    result = openwebui._knowledge_tide_file(
        server.client,
        "kb-1",
        f"rules/{server.uuid}/query.spl",
        cache,
    )
    assert result is not None
    assert result[2] == server.contents["query"]
    assert any(
        req.get("directory_id") == "rule" and req.get("query") == "query.spl"
        for req in server.requests
    )


def _openwebui_stream_test_state():
    return {
        "answer": "",
        "reasoning_blocks": [],
        "tool_done": {},
        "tool_results": [],
        "phase": "initial model",
        "phase_started_at": 0.0,
        "last_progress_at": 0.0,
        "stream_started": False,
        "stream_delta_count": 0,
    }


def test_openwebui_response_completion_streams_tokens_before_final_snapshot():
    state = _openwebui_stream_test_state()
    chunks = []
    statuses = []
    def callback(data):
        return openwebui_socket._consume_response_completion(
            data,
            state,
            on_text=chunks.append,
            on_reasoning=lambda value: None,
            on_tool=lambda value: None,
            on_status=statuses.append,
        )
    assert callback({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "delta": "Hello",
    })
    assert chunks == ["Hello"]
    assert state["answer"] == "Hello"
    assert callback({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "delta": " world",
    })
    assert chunks == ["Hello", " world"]
    assert state["stream_delta_count"] == 2
    assert statuses == ["live model token stream started"]

    # Periodic/final chat:completion snapshots are authoritative but must not
    # repeat tokens that the real-time stream already displayed.
    openwebui_socket._reconcile_answer_snapshot("Hello world", state, chunks.append)
    assert chunks == ["Hello", " world"]
    openwebui_socket._reconcile_answer_snapshot("Hello world!", state, chunks.append)
    assert chunks == ["Hello", " world", "!"]
    assert state["answer"] == "Hello world!"


def test_openwebui_response_completion_streams_reasoning_in_order():
    state = _openwebui_stream_test_state()
    chunks = []
    def callback(data):
        return openwebui_socket._consume_response_completion(
            data,
            state,
            on_text=lambda value: None,
            on_reasoning=chunks.append,
            on_tool=lambda value: None,
            on_status=lambda value: None,
        )
    for fragment in ("Check ", "the logs"):
        assert callback({
            "type": "response.reasoning_text.delta",
            "item_id": "reasoning_1",
            "output_index": 0,
            "delta": fragment,
        })
    assert chunks == ["Check ", "the logs"]
    assert state["reasoning_blocks"] == ["Check the logs"]
    assert state["phase"] == "model reasoning"
    assert state["stream_started"] is True

    # Native Responses API reasoning summary messages use their own delta type.
    assert callback({
        "type": "response.reasoning_summary_text.delta",
        "item_id": "reasoning_2",
        "output_index": 1,
        "delta": "Next step",
    })
    assert chunks[-2:] == ["\n\n", "Next step"]
    assert state["reasoning_blocks"] == ["Check the logs", "Next step"]


def test_openwebui_response_completion_tool_start_and_result_are_live():
    state = _openwebui_stream_test_state()
    activity = []
    statuses = []
    def callback(data):
        return openwebui_socket._consume_response_completion(
            data,
            state,
            on_text=lambda value: None,
            on_reasoning=lambda value: None,
            on_tool=activity.append,
            on_status=statuses.append,
        )
    item = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "splunk_search",
    }
    assert callback({
        "type": "response.output_item.added",
        "output_index": 1,
        "item": item,
    })
    assert state["tool_done"]["call_1"] is False
    assert activity == ["↳ splunk_search ..."]
    assert callback({
        "type": "response.function_call_arguments.delta",
        "item_id": "call_1",
        "delta": '{"search": "index=aws"}',
    })
    assert state["phase"] == "preparing tool call"
    assert callback({
        "type": "response.output_item.done",
        "output_index": 1,
        "item": item,
    })
    assert state["tool_done"]["call_1"] is False
    assert state["phase"] == "tool execution"
    assert callback({
        "type": "response.output_item.done",
        "output_index": 2,
        "item": {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "1 event found",
        },
    })
    assert activity == ["↳ splunk_search ...", "↳ splunk_search done"]
    assert state["tool_results"] == [
        {"name": "splunk_search", "result": "1 event found"}
    ]
    assert state["phase"] == "model continuation"
    assert statuses == ["tool result received; waiting for model continuation"]


def test_openwebui_socket_runner_consumes_live_events_before_final(monkeypatch):
    import asyncio

    events = []
    payloads = [
        {
            "type": "chat:completion",
            "data": {
                "sources": [{
                    "source": {"name": "splunk-mcp_search"},
                    "document": ["{'event_count': 8, 'index': 'digit_sec'}"],
                    "metadata": [{
                        "source": "splunk-mcp_search",
                        "parameters": {"query": "index=digit_sec | stats count"},
                    }],
                    "tool_result": True,
                }],
            },
        },
        {
            "type": "response:completion",
            "data": {
                "type": "response.reasoning_text.delta",
                "item_id": "reasoning_1",
                "output_index": 0,
                "delta": "Thinking",
            },
        },
        {
            "type": "response:completion",
            "data": {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 1,
                "delta": "Hello",
            },
        },
        {
            "type": "chat:completion",
            "data": {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 1,
                "delta": " world",
            },
        },
        {
            "type": "chat:completion",
            "data": {
                "done": True,
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Thinking"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello world"}],
                    },
                ],
            },
        },
    ]

    class FakeHTTPClientSession:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is True

        async def close(self):
            pass

    class FakeSocket:
        def __init__(self, http_session):
            self.handlers = {}

        def on(self, event, handler):
            self.handlers[event] = handler

        async def connect(self, url, **kwargs):
            pass

        def get_sid(self, namespace):
            return "socket-1"

        async def call(self, method, payload, timeout):
            assert method == "user-join"

            async def push_stream():
                for event in payloads:
                    await asyncio.sleep(0)
                    await self.handlers["events"]({
                        "chat_id": "temporary:socket-1",
                        "data": event,
                    })
            asyncio.create_task(push_stream())
            return {"id": "test-user"}

        async def emit(self, event, payload):
            pass

        async def disconnect(self):
            pass

    monkeypatch.setattr(
        openwebui_socket,
        "_need_socketio",
        lambda: SimpleNamespace(AsyncClient=FakeSocket),
    )
    monkeypatch.setattr("aiohttp.ClientSession", FakeHTTPClientSession)
    monkeypatch.setattr(
        openwebui_socket.http,
        "json_request",
        lambda *args, **kwargs: {
            "status": True,
            "task_ids": ["task-1"],
            "chat_id": "temporary:socket-1",
        },
    )

    async def run():
        return await openwebui_socket.run_chat_with_tools_with_files(
            base_url="https://example.test",
            token="test-jwt",
            model="glm-5.3",
            messages=[{"role": "user", "content": "Hello"}],
            tool_ids=["server:mcp:splunk-mcp"],
            timeout=10,
            on_text=lambda value: events.append(("text", value)),
            on_reasoning=lambda value: events.append(("reasoning", value)),
            on_status=lambda value: events.append(("status", value)),
            on_source=lambda value: events.append(("source", value)),
            on_tool=lambda value: events.append(("tool", value)),
        )

    result = asyncio.run(run())
    assert result["answer"] == "Hello world"
    assert result["reasoning"] == "Thinking"
    assert result["remote_task_ids"] == ["task-1"]
    assert [value for kind, value in events if kind == "text"] == [
        "Hello", " world"
    ]
    assert [value for kind, value in events if kind == "reasoning"] == [
        "Thinking"
    ]
    assert ("status", "live model token stream started") in events
    assert len(result["server_tool_sources"]) == 1
    provenance = result["server_tool_sources"][0]
    assert provenance["is_mcp"] is True
    assert provenance["server_executed"] is True
    assert provenance["source_id"] == 1
    assert provenance["parameters"]["query"] == "index=digit_sec | stats count"
    assert "event_count" in provenance["result_excerpt"]
    assert ("source", provenance) in events
    assert any(kind == "tool" and "splunk-mcp_search" in line for kind, line in events)



def test_legacy_mcp_source_capture_matches_openwebui_numbered_source_ids():
    state = _openwebui_stream_test_state()
    state["last_progress_at"] = 0.0
    captured = []
    activity = []
    sources = [
        {
            "source": {"id": "file-1", "name": "Knowledge file"},
            "document": ["unrelated KB metadata"],
            "metadata": [{"source": "file-1"}],
        },
        {
            "source": {"name": "splunk-mcp_search"},
            "document": ["{'event_count': 8, 'et': 1791286913.205}"],
            "metadata": [{
                "source": "splunk-mcp_search",
                "parameters": {"query": "index=digit_sec | stats count"},
            }],
            "tool_result": True,
        },
    ]
    args = {
        "tool_ids": ["server:mcp:splunk-mcp"],
        "on_tool": activity.append,
        "on_source": captured.append,
    }
    assert openwebui_socket._capture_server_tool_sources(sources, state, **args) == 1
    assert captured[0]["source_id"] == 2
    assert captured[0]["is_mcp"] is True
    assert captured[0]["server_executed"] is True
    assert captured[0]["parameters"]["query"] == "index=digit_sec | stats count"
    assert "1791286913.205" in captured[0]["result_excerpt"]
    assert captured[0]["result_truncated"] is False
    assert len(captured[0]["result_sha256"]) == 64
    assert state["tool_results"][0]["source_id"] == 2
    assert "splunk-mcp_search" in activity[0]
    # Replayed source snapshots must not inflate the evidence count.
    assert openwebui_socket._capture_server_tool_sources(sources, state, **args) == 0
    assert len(captured) == 1
    assert len(activity) == 1


def test_legacy_mcp_source_capture_ignores_non_tool_citations():
    state = _openwebui_stream_test_state()
    captured = []
    assert openwebui_socket._capture_server_tool_sources(
        [{
            "source": {"name": "KB excerpt"},
            "metadata": [{"source": "kb-1"}],
            "document": ["one fake source"],
        }],
        state,
        tool_ids=["server:mcp:splunk-mcp"],
        on_tool=lambda value: None,
        on_source=captured.append,
    ) == 0
    assert captured == []
    assert state["tool_results"] == []


@pytest.mark.usefixtures("supported_openwebui_server")
@pytest.mark.parametrize(
    "tool_calls, expected_error",
    [
        ([], True),
        ([{"name": "splunk-mcp_search", "result": "8 events"}], False),
    ],
)
def test_explicit_mcp_search_requires_server_executed_evidence(
    monkeypatch, tool_calls, expected_error
):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "models": [],
            "enabled_tool_ids": ["server:mcp:splunk-mcp"],
        },
    )

    class FakeClient:
        base_url = "https://example.test"
        token = "jwt"
        timeout = 1200

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda client, model_id: {"id": model_id, "name": "GLM"},
    )
    monkeypatch.setattr(
        openwebui,
        "_prepare_openwebui_request",
        lambda prompt, client, **kwargs: (
            [{"role": "user", "content": "Execute Splunk MCP searches for this notable"}],
            [],
        ),
    )
    async def fake_runner(**kwargs):
        return {
            "answer": "8 events",
            "reasoning": None,
            "tool_calls": tool_calls,
            "raw_content": "8 events",
            "server_tool_sources": [],
            "remote_chat_id": "temporary:test",
            "remote_task_ids": ["task"],
        }
    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", fake_runner)
    model = openwebui.OpenWebUIModel("glm-5.3")
    result = SimpleNamespace(response_json=None)
    if expected_error:
        with pytest.raises(llm.ModelError, match="No server-executed MCP search"):
            list(model.execute(llm.Prompt("Investigate", model), True, result, None))
    else:
        list(model.execute(llm.Prompt("Investigate", model), True, result, None))
        assert result.response_json["verified_mcp_tool_results"] == 0
        assert result.response_json["tool_calls"] == tool_calls


def test_native_mcp_transport_requires_server_disabled_builtins():
    model = {"id": "glm-5.3", "info": {"meta": {"capabilities": {}}}}
    with pytest.raises(llm.ModelError, match="builtin_tools=false"):
        openwebui._require_isolated_native_mcp(model)
    model["info"]["meta"]["capabilities"]["builtin_tools"] = True
    with pytest.raises(llm.ModelError, match="builtin_tools=false"):
        openwebui._require_isolated_native_mcp(model)
    model["info"]["meta"]["capabilities"]["builtin_tools"] = False
    openwebui._require_isolated_native_mcp(model)


def test_native_mcp_transport_refuses_direct_model_metadata_override():
    model = {
        "id": "glm-5.3",
        "direct": True,
        "info": {"meta": {"capabilities": {"builtin_tools": False}}},
    }
    with pytest.raises(llm.ModelError, match="builtin_tools=false"):
        openwebui._require_isolated_native_mcp(model)


@pytest.mark.usefixtures("supported_openwebui_server")
def test_sessionless_native_rejected_before_tool_submission(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "enabled_tool_ids": ["server:mcp:splunk-mcp"],
        },
    )

    class FakeClient:
        base_url = "https://example.test"
        token = "jwt"
        timeout = 1200

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
    monkeypatch.setattr(
        openwebui, "_get_model_item", lambda client, model_id: {"id": model_id}
    )
    monkeypatch.setattr(
        openwebui,
        "_prepare_openwebui_request",
        lambda *args, **kwargs: ([{"role": "user", "content": "investigate"}], []),
    )
    monkeypatch.setattr(openwebui, "_enabled_knowledge_items", lambda *args: [])
    model = openwebui.OpenWebUIModel("glm-5.3")
    prompt = llm.Prompt(
        "investigate",
        model,
        options=model.Options(openwebui_mcp_transport="sessionless_native"),
    )
    with pytest.raises(
        llm.ModelError,
        match="cannot run Open WebUI's iterative",
    ):
        list(model.execute(prompt, True, SimpleNamespace(response_json=None), None))


def test_default_openwebui_mcp_transport_is_safe_legacy():
    model = openwebui.OpenWebUIModel("glm-5.3")
    assert model.Options().openwebui_mcp_transport == "background_legacy"


@pytest.mark.parametrize(
    "builtin_tools, expected_ready",
    [(True, False), (None, False), (False, True)],
)
def test_openwebui_doctor_reports_native_mcp_readiness(
    monkeypatch, builtin_tools, expected_ready
):
    from click.testing import CliRunner

    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {"enabled_tool_ids": ["server:mcp:splunk-mcp"]},
    )
    monkeypatch.setattr(openwebui, "_client", lambda config: object())
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda client, model_id: {
            "id": model_id,
            "info": {"meta": {"capabilities": {"builtin_tools": builtin_tools}}},
        },
    )

    @click.group()
    def app():
        pass

    openwebui.register_commands(app)
    result = CliRunner().invoke(
        app, ["openwebui", "doctor", "--model", "glm-5.3"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["iterative_mcp_ready"] is expected_ready
    assert payload["server_builtin_tools"] is builtin_tools
    assert payload["selected_tool_ids"] == ["server:mcp:splunk-mcp"]
    assert payload["recommended_transport"] == (
        "background_native" if expected_ready else "background_legacy (one-shot)"
    )



def test_native_openwebui_sequential_mcp_results_are_both_captured():
    state = _openwebui_stream_test_state()
    activity = []

    def receive(payload):
        return openwebui_socket._consume_response_completion(
            payload,
            state,
            on_text=lambda value: None,
            on_reasoning=lambda value: None,
            on_tool=activity.append,
            on_status=lambda value: None,
        )

    for index, (call_id, count) in enumerate(
        (("call-guardduty", 2), ("call-cloudtrail", 3)), start=1
    ):
        assert receive(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "splunk-mcp_splunk_run_query",
                },
            }
        )
        assert receive(
            {
                "type": "response.output_item.done",
                "output_index": index,
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": {"results": [{"count": count}]},
                },
            }
        )

    assert state["tool_done"] == {
        "call-guardduty": True,
        "call-cloudtrail": True,
    }
    assert len(state["tool_results"]) == 2
    counts = [
        item["result"]["results"][0]["count"]
        for item in state["tool_results"]
    ]
    assert counts == [2, 3]
    assert len([item for item in activity if " done" in item]) == 2
