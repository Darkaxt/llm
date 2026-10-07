import io
import zipfile
import json

import pytest
import click
from types import SimpleNamespace

import llm
from llm.default_plugins import openwebui, openwebui_socket
from llm.parts import AttachmentPart, Message, TextPart


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

    model = openwebui.OpenWebUIModel("glm-5.3")
    prompt = llm.Prompt("test", model)
    response = SimpleNamespace(response_json=None)

    chunks = list(model.execute(prompt, True, response, None))
    assert chunks[0].type == "reasoning"
    assert chunks[0].chunk == "thinking"
    assert chunks[1:] == ["hello", " world"]
    assert response.response_json["remote_model"] == "glm-5.3"
    assert response.response_json["tool_activity"] == ["splunk_search"]


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
    assert openwebui._chat_timeout() == 600

    monkeypatch.setenv("LLM_OPENWEBUI_CHAT_TIMEOUT", "900")
    assert openwebui._chat_timeout() == 900



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
    assert list(model.execute(prompt, True, response, None)) == []
    assert seen["extra"] == ["server:mcp:splunk"]


def test_resolve_tool_selector_matches_mcp_name():
    tools = [
        SimpleNamespace(id="server:mcp:splunk-main", name="Splunk MCP"),
        SimpleNamespace(id="local-tool", name="Local Tool"),
    ]
    client = SimpleNamespace(list_tools=lambda: tools)

    tool = openwebui._resolve_tool_selector(client, "Splunk MCP")
    assert tool.id == "server:mcp:splunk-main"
    assert openwebui._tool_kind(tool.id) == "mcp"



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


def test_enabled_knowledge_items_use_minimal_collection_shape(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_get_knowledge_by_id",
        lambda client, knowledge_id: {
            "id": knowledge_id,
            "name": "TIDE Splunk Investigation",
            "description": "Reusable TIDE knowledge",
            "files": [{"id": "huge-file-list-entry"}],
            "access_grants": [{"permission": "read"}],
        },
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
        }
    ]


def test_enabled_knowledge_is_added_to_chat_request(monkeypatch):
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "models": [],
            "enabled_knowledge_ids": ["kb-1"],
        },
    )

    calls = []

    class FakeClient:
        base_url = "https://example.test"
        token = "jwt"
        timeout = 600

        def resolve_tools(self, model_id, extra_tool_ids=None, no_tools=False):
            return []

        def run_chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                answer="ok",
                reasoning=None,
                tool_calls=[],
                raw_content="ok",
            )

    monkeypatch.setattr(openwebui, "_client", lambda config: FakeClient())
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
        lambda config, client: [
            {
                "type": "collection",
                "id": "kb-1",
                "name": "TIDE Splunk Investigation",
                "description": "Reusable TIDE knowledge",
            }
        ],
    )

    model = openwebui.OpenWebUIModel("glm-5.3")
    prompt = llm.Prompt("test", model)
    response = SimpleNamespace(response_json=None)

    assert list(model.execute(prompt, True, response, None)) == []
    assert len(calls) == 1
    assert calls[0]["extra"]["files"] == [
        {
            "type": "collection",
            "id": "kb-1",
            "name": "TIDE Splunk Investigation",
            "description": "Reusable TIDE knowledge",
        }
    ]


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



def test_resume_knowledge_file_reuses_oldest_matching_upload(monkeypatch):
    calls = []

    candidates_by_hash = {
        "abc": [
            {
                "id": "old-file",
                "filename": "rule-index.json",
                "created_at": 10,
                "data": {"status": "completed"},
                "meta": {
                    "name": "rule-index.json",
                    "file_hash": "abc",
                    "data": {
                        "knowledge_id": "kb-1",
                        "directory_id": "dir-1",
                    },
                },
            },
            {
                "id": "new-failed-retry",
                "filename": "rule-index.json",
                "created_at": 20,
                "data": {
                    "status": "failed",
                    "error": "Duplicate content detected.",
                },
                "meta": {
                    "name": "rule-index.json",
                    "file_hash": "abc",
                    "data": {
                        "knowledge_id": "kb-1",
                        "directory_id": "dir-1",
                    },
                },
            },
        ]
    }

    def fake_json(client, method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path.endswith("/file/add"):
            assert kwargs["json_body"]["file_id"] == "old-file"
            return {"id": "kb-1"}
        raise AssertionError(path)

    monkeypatch.setattr(openwebui, "_owui_http_json", fake_json)

    entry = {
        "filename": "rule-index.json",
        "path": "bundle",
        "checksum": "abc",
    }
    client = SimpleNamespace(timeout=600)

    assert openwebui._resume_knowledge_file(
        client,
        knowledge_id="kb-1",
        entry=entry,
        directory_id="dir-1",
        candidates_by_hash=candidates_by_hash,
    )
    assert len(calls) == 1


def test_matching_resume_candidates_requires_filename_and_prefers_directory():
    candidates_by_hash = {
        "abc": [
            {
                "id": "wrong-name",
                "filename": "macros.json",
                "meta": {
                    "name": "macros.json",
                    "data": {"directory_id": "dir-1"},
                },
            },
            {
                "id": "fallback-root",
                "filename": "rule-index.json",
                "meta": {
                    "name": "rule-index.json",
                    "data": {},
                },
            },
            {
                "id": "exact",
                "filename": "rule-index.json",
                "meta": {
                    "name": "rule-index.json",
                    "data": {"directory_id": "dir-1"},
                },
            },
        ]
    }

    matches = openwebui._matching_resume_candidates(
        {
            "filename": "rule-index.json",
            "checksum": "abc",
        },
        "dir-1",
        candidates_by_hash,
    )
    assert [item["id"] for item in matches] == ["exact", "fallback-root"]
