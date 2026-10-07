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
        lambda client, attachment, index: calls.append(index) or uploaded,
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
        def resolve_tools(self, model_id, no_tools=False):
            assert model_id == "glm-5.3"
            assert no_tools is False
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
