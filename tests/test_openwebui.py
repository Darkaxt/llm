import json
from types import SimpleNamespace

import llm
from llm.default_plugins import openwebui
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
        lambda prompt, client: (
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
        lambda prompt, client: ([{"role": "user", "content": "test"}], []),
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
