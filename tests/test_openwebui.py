import json
from types import SimpleNamespace

import pytest

import llm
from llm.default_plugins import openwebui
from llm.parts import Message, TextPart


def test_messages_for_openwebui_uses_full_chain():
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
    assert openwebui._messages_for_openwebui(prompt) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]


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
