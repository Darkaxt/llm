import json
import zipfile

import pytest
from click.testing import CliRunner
from pytest_httpx2 import IteratorStream

import llm
from llm.cli import cli

BASE = "https://gateway.example.test/v1"
MODEL = "test-investigator"


def create_profile(user_path):
    result = CliRunner().invoke(
        cli, ["apisix", "profile", "create", "work", "--base-url", BASE]
    )
    assert result.exit_code == 0, result.output
    (user_path / "keys.json").write_text(
        json.dumps({"apisix-work": "test-secret"}), encoding="utf-8"
    )


def completion(text="APISIX_PROBE_OK", calls=None):
    return {
        "id": "probe",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "tool_calls": calls},
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def stream(deltas, finish="stop"):
    for delta in [*deltas, {}]:
        yield (
            "data: "
            + json.dumps(
                {
                    "id": "probe",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish if not delta else None,
                        }
                    ],
                }
            )
            + "\n\n"
        ).encode()
    yield b"data: [DONE]\n\n"


def add_stream(mock, deltas, finish="stop"):
    mock.add_response(
        method="POST",
        url=BASE + "/chat/completions",
        stream=IteratorStream(stream(deltas, finish)),
        headers={"Content-Type": "text/event-stream"},
    )


def test_profile_creation_preserves_existing_state(user_path):
    marker = user_path / "openwebui.json"
    marker.write_text("preserve", encoding="utf-8")
    create_profile(user_path)
    config = json.loads((user_path / "apisix.json").read_text(encoding="utf-8"))
    assert config["profiles"]["work"]["key_alias"] == "apisix-work"
    assert "test-secret" not in json.dumps(config)
    result = CliRunner().invoke(
        cli, ["apisix", "profile", "create", "work", "--base-url", BASE]
    )
    assert result.exit_code != 0
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_profile_tls_opt_out_is_a_creation_flag(user_path):
    result = CliRunner().invoke(
        cli,
        ["apisix", "profile", "create", "another", "--no-verify-ssl"],
        input=BASE + "\n",
    )
    assert result.exit_code == 0, result.output
    config = json.loads((user_path / "apisix.json").read_text(encoding="utf-8"))
    assert config["profiles"]["another"]["verify_ssl"] is False


def test_discovery_registers_actual_model(user_path, httpx2_mock):
    create_profile(user_path)
    httpx2_mock.add_response(
        method="GET", url=BASE + "/models", json={"data": [{"id": MODEL}]}
    )
    result = CliRunner().invoke(cli, ["apisix", "models", "--profile", "work"])
    assert result.exit_code == 0, result.output
    assert "apisix/work/" + MODEL in result.output
    request = httpx2_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer test-secret"


def test_all_availability_probes_and_safe_report(user_path, tmp_path, httpx2_mock):
    create_profile(user_path)
    httpx2_mock.add_response(
        method="GET", url=BASE + "/models", json={"data": [{"id": MODEL}]}
    )
    httpx2_mock.add_response(
        method="POST", url=BASE + "/chat/completions", json=completion()
    )
    add_stream(httpx2_mock, [{"content": "APISIX_"}, {"content": "PROBE_OK"}])
    add_stream(
        httpx2_mock,
        [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "probe-call",
                        "type": "function",
                        "function": {
                            "name": "availability_echo",
                            "arguments": '{"value":',
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '"APISIX_TOOL_OK"}'}}
                ]
            },
        ],
        "tool_calls",
    )
    httpx2_mock.add_response(
        method="POST", url=BASE + "/chat/completions", json=completion("APISIX_TOOL_OK")
    )
    output = tmp_path / "availability.json"
    result = CliRunner().invoke(
        cli,
        [
            "apisix",
            "check",
            "--profile",
            "work",
            "--model",
            MODEL,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert all(
        report["checks"][k]["status"] == "passed"
        for k in [
            "models",
            "completion",
            "streaming",
            "function_call",
            "tool_continuation",
        ]
    )
    assert report["checks"]["splunk_mcp"]["status"] == "not_tested"
    assert report["checks"]["streaming"]["text_deltas"] == 2
    assert "test-secret" not in output.read_text()
    assert "gateway.example.test" not in output.read_text()
    requests = httpx2_mock.get_requests()
    for request in requests[1:]:
        body = json.loads(request.content)
        assert "reasoning_effort" not in body
        assert all(value is None for value in request.extensions["timeout"].values())
    assert json.loads(requests[-1].content)["messages"][-1]["role"] == "tool"


def test_auth_failure_is_reported_without_leaking_error_body(
    user_path, tmp_path, httpx2_mock
):
    create_profile(user_path)
    httpx2_mock.add_response(
        method="GET",
        url=BASE + "/models",
        status_code=401,
        json={"error": {"message": "test-secret " + BASE}},
    )
    output = tmp_path / "availability.json"
    result = CliRunner().invoke(
        cli,
        [
            "apisix",
            "check",
            "--profile",
            "work",
            "--model",
            MODEL,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    report = json.loads(output.read_text())
    assert report["checks"]["models"]["http_status"] == 401
    assert report["checks"]["completion"]["status"] == "not_tested"
    assert "test-secret" not in result.output + output.read_text()
    assert BASE not in result.output + output.read_text()


def test_existing_archive_knowledge_is_reusable_without_upload(user_path, tmp_path):
    from llm.default_plugins.apisix import inspect_knowledge

    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "SKILL.md").write_text("Investigate read-only telemetry.")
    (root / "SPEC.md").write_text("Follow the evidence.")
    with zipfile.ZipFile(root / "existing.zip", "w") as archive:
        for name, content in {
            "detection.json": "{}",
            "query.spl": "index=example",
            "dependencies.json": '{"path":"macros/by-stanza/source.json","resolution_policy":"resolve"}',
        }.items():
            archive.writestr(
                "bundle/rules/00000000-0000-0000-0000-000000000001/" + name, content
            )
        archive.writestr(
            "bundle/macros/by-stanza/source.json", '{"definition":"index=example"}'
        )
    report = inspect_knowledge(str(root))
    assert report["status"] == "reusable"
    assert report["rules"] == 1
    assert report["macro_files"] == 1
    assert str(root) not in json.dumps(report)


def test_missing_macro_prevents_reuse_claim(tmp_path):
    from llm.default_plugins.apisix import inspect_knowledge

    (tmp_path / "SKILL.md").write_text("skill")
    (tmp_path / "SPEC.md").write_text("spec")
    rule = tmp_path / "rules" / "one"
    rule.mkdir(parents=True)
    (rule / "detection.json").write_text("{}")
    (rule / "query.spl").write_text("index=example")
    (rule / "dependencies.json").write_text(
        '{"path":"macros/by-stanza/missing.json","resolution_policy":"resolve"}'
    )
    assert inspect_knowledge(str(tmp_path))["status"] == "incomplete"


def test_normal_client_tool_loop(user_path, httpx2_mock):
    create_profile(user_path)
    from llm.default_plugins.apisix import _cache_models

    _cache_models("work", [MODEL])
    model = llm.get_model("apisix/work/" + MODEL)

    def availability_echo(value: str) -> str:
        "Return the supplied synthetic marker."
        return value

    add_stream(
        httpx2_mock,
        [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "local-call",
                        "type": "function",
                        "function": {
                            "name": "availability_echo",
                            "arguments": '{"value":"APISIX_TOOL_OK"}',
                        },
                    }
                ]
            }
        ],
        "tool_calls",
    )
    add_stream(httpx2_mock, [{"content": "APISIX_TOOL_OK"}])
    chain = model.chain("Call the synthetic tool", tools=[availability_echo])
    assert chain.text() == "APISIX_TOOL_OK"
    requests = httpx2_mock.get_requests()
    assert (
        json.loads(requests[1].content)["messages"][-1]["content"] == "APISIX_TOOL_OK"
    )
    assert all(
        "reasoning_effort" not in json.loads(request.content) for request in requests
    )


@pytest.mark.parametrize("verify", [True, False])
def test_tls_policy_is_explicit_and_profile_scoped(user_path, monkeypatch, verify):
    from llm.default_plugins.apisix import _client, _profile

    args = ["apisix", "profile", "create", "work", "--base-url", BASE]
    if not verify:
        args.append("--no-verify-ssl")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    (user_path / "keys.json").write_text(
        '{"apisix-work":"test-secret"}', encoding="utf-8"
    )
    observed = {}
    import httpx2

    original = httpx2.Client.__init__

    def capture(self, **kwargs):
        observed.update(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(httpx2.Client, "__init__", capture)
    with _client("work", _profile("work")) as client:
        assert client.max_retries == 0
        assert client.timeout is None
    assert observed == {"verify": verify, "timeout": None, "follow_redirects": False}


@pytest.mark.parametrize("payload", [{"data": None}, {"data": [None]}])
def test_unknown_or_missing_models_is_a_failed_probe(
    user_path, tmp_path, httpx2_mock, payload
):
    create_profile(user_path)
    httpx2_mock.add_response(method="GET", url=BASE + "/models", json=payload)
    httpx2_mock.add_response(
        method="POST", url=BASE + "/chat/completions", json=completion()
    )
    add_stream(httpx2_mock, [{"content": "APISIX_PROBE_OK"}])
    add_stream(httpx2_mock, [{"content": "I cannot call tools"}])
    output = tmp_path / "invalid.json"
    result = CliRunner().invoke(
        cli,
        [
            "apisix",
            "check",
            "--profile",
            "work",
            "--model",
            MODEL,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    report = json.loads(output.read_text())
    assert report["checks"]["models"]["status"] == "failed"
    assert report["checks"]["function_call"]["status"] == "failed"
    assert report["checks"]["tool_continuation"]["status"] == "not_tested"


def test_no_models_registered_before_discovery(user_path):
    create_profile(user_path)
    with pytest.raises(llm.UnknownModelError):
        llm.get_model("apisix/work/" + MODEL)


def test_single_delta_does_not_prove_incremental_streaming(user_path, httpx2_mock):
    from llm.default_plugins.apisix import _client, _profile, _stream_probe

    create_profile(user_path)
    add_stream(httpx2_mock, [{"content": "APISIX_PROBE_OK"}])
    with _client("work", _profile("work")) as client:
        result, _ = _stream_probe(client, MODEL, [{"role": "user", "content": "probe"}])
    assert result["status"] == "failed"
    assert result["incremental_observed"] is False


def test_malformed_completion_still_writes_safe_report(
    user_path, tmp_path, httpx2_mock
):
    create_profile(user_path)
    httpx2_mock.add_response(
        method="GET", url=BASE + "/models", json={"data": [{"id": MODEL}]}
    )
    response = completion()
    response["choices"][0]["message"] = None
    httpx2_mock.add_response(
        method="POST", url=BASE + "/chat/completions", json=response
    )
    add_stream(httpx2_mock, [{"content": "APISIX_"}, {"content": "PROBE_OK"}])
    add_stream(httpx2_mock, [{"content": "No tools"}])
    output = tmp_path / "malformed.json"
    result = CliRunner().invoke(
        cli,
        [
            "apisix",
            "check",
            "--profile",
            "work",
            "--model",
            MODEL,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert (
        json.loads(output.read_text())["checks"]["completion"]["error_type"]
        == "AttributeError"
    )
