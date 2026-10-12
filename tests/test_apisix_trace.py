import base64
import gzip
import json

import httpx2
import pytest
from click.testing import CliRunner
from pytest_httpx2 import IteratorStream
from test_apisix import BASE, MODEL, create_profile, stream

import llm
from llm.cli import cli
from llm.default_plugins.apisix import ApisixChat


def records(directory):
    paths = list(directory.glob("*.jsonl"))
    assert len(paths) == 1
    return paths[0], [json.loads(line) for line in paths[0].read_text().splitlines()]


@pytest.mark.parametrize("compressed", [False, True])
def test_capture_before_sdk_filtering(
    user_path, tmp_path, httpx2_mock, monkeypatch, compressed
):
    create_profile(user_path)
    directory = tmp_path / "capture"
    monkeypatch.setenv("LLM_APISIX_TRACE_DIR", str(directory))
    body = b": heartbeat\r\n\r\n" + b"".join(
        stream(
            [
                {
                    "reasoning_content": "Thinking: \u03bb",
                    "future_field": {"keep": True},
                },
                {"reasoning": "More thinking"},
                {"content": "Finished"},
            ]
        )
    )
    raw = gzip.compress(body) if compressed else body
    parts = [raw[:9], raw[9:40], raw[40:]]
    headers = {
        "Content-Type": "text/event-stream",
        "X-Request-ID": "correlation-1",
        "Set-Cookie": "private-cookie",
    }
    if compressed:
        headers["Content-Encoding"] = "gzip"
    httpx2_mock.add_response(
        method="POST",
        url=BASE + "/chat/completions",
        stream=IteratorStream(parts),
        headers=headers,
    )
    model = ApisixChat("work", {"base_url": BASE, "key_alias": "apisix-work"}, MODEL)
    assert model.prompt("Synthetic test").text() == "Finished"
    path, rows = records(directory)
    assert (
        b"".join(
            base64.b64decode(row["body_b64"]) for row in rows if row["type"] == "chunk"
        )
        == raw
    )
    request = json.loads(base64.b64decode(rows[0]["body_b64"]))
    assert request["messages"][-1]["content"] == "Synthetic test"
    assert "reasoning_effort" not in request
    assert all(
        value is None
        for value in httpx2_mock.get_requests()[0].extensions["timeout"].values()
    )
    assert "test-secret" not in path.read_text()
    assert "private-cookie" not in path.read_text()
    assert "gateway.example.test" not in path.read_text()
    assert all(row["elapsed_seconds"] >= 0 for row in rows)
    result = CliRunner().invoke(cli, ["apisix", "trace-summary", str(path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["reasoning_events"] == 2
    assert report["answer_events"] == 1
    assert report["done_received"] is True
    assert report["finish_reasons"] == ["stop"]
    assert report["upstream_request_id"] == "correlation-1"
    assert "Thinking" not in result.output
    assert "Synthetic test" not in result.output


def test_capture_disabled_by_default(user_path, tmp_path, httpx2_mock, monkeypatch):
    create_profile(user_path)
    monkeypatch.delenv("LLM_APISIX_TRACE_DIR", raising=False)
    httpx2_mock.add_response(
        method="POST",
        url=BASE + "/chat/completions",
        stream=IteratorStream(stream([{"content": "OK"}])),
        headers={"Content-Type": "text/event-stream"},
    )
    assert (
        ApisixChat("work", {"base_url": BASE, "key_alias": "apisix-work"}, MODEL)
        .prompt("Test")
        .text()
        == "OK"
    )
    assert not list(tmp_path.glob("**/*trace*.jsonl"))


@pytest.mark.parametrize("failure", [KeyboardInterrupt, httpx2.ReadError])
def test_capture_error_and_cancellation(
    user_path, tmp_path, httpx2_mock, monkeypatch, failure
):
    from llm.apisix_trace import summarize_trace

    create_profile(user_path)
    monkeypatch.setenv("LLM_APISIX_TRACE_DIR", str(tmp_path / "capture"))

    class Broken(httpx2.SyncByteStream):
        def __iter__(self):
            yield b": alive\n\n"
            raise failure("private diagnostic message")

    httpx2_mock.add_response(
        method="POST",
        url=BASE + "/chat/completions",
        headers={"Content-Type": "text/event-stream"},
        stream=Broken(),
    )
    expected = KeyboardInterrupt if failure is KeyboardInterrupt else llm.ModelError
    with pytest.raises(expected):
        ApisixChat(
            "work", {"base_url": BASE, "key_alias": "apisix-work"}, MODEL
        ).prompt("Synthetic cancellation test").text()
    path, rows = records(tmp_path / "capture")
    assert rows[-1]["outcome"] == (
        "cancelled" if failure is KeyboardInterrupt else "error"
    )
    assert "private diagnostic message" not in path.read_text()
    report = summarize_trace(path)
    assert report["done_received"] is False
    assert report["eof_observed"] is False


def test_timing_fragmented_frames_and_truncation(tmp_path):
    from llm.apisix_trace import summarize_trace

    payload = b"".join(
        stream(
            [
                {"reasoning_content": "one"},
                {"reasoning_content": "two"},
                {"content": "answer"},
            ]
        )
    )
    frames = payload.split(b"\n\n")
    chunks = [
        (2, frames[0] + b"\r"),
        (3, b"\n\r\n"),
        (310, frames[1] + b"\n\n"),
        (315, frames[2] + b"\n\n"),
        (320, b'data: {"incomplete":'),
    ]
    rows = [
        {
            "type": "request",
            "elapsed_seconds": 0,
            "started_utc": "2026-10-12T12:00:00+00:00",
            "trace_id": "test",
        },
        {
            "type": "response",
            "elapsed_seconds": 1,
            "status_code": 200,
            "content_type": "text/event-stream",
        },
    ]
    rows += [
        {
            "type": "chunk",
            "elapsed_seconds": elapsed,
            "body_b64": base64.b64encode(body).decode(),
        }
        for elapsed, body in chunks
    ]
    rows += [{"type": "end", "elapsed_seconds": 325, "outcome": "eof"}]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report = summarize_trace(path)
    assert report["duration_seconds"] == 325
    assert report["first_byte_seconds"] == 2
    assert report["first_reasoning_seconds"] == 3
    assert report["first_answer_seconds"] == 315
    assert report["reasoning_event_span_seconds"] == 307
    assert report["max_inter_reasoning_event_gap_seconds"] == 307
    assert report["max_no_byte_gap_seconds"] == 307
    assert report["eof_observed"] is True
    assert report["done_received"] is False
    assert report["incomplete_sse_tail"] is True
    assert report["finish_reasons"] == []


def test_connection_failure_records_duration_without_private_error(tmp_path):
    from llm.apisix_trace import CaptureClient, summarize_trace

    def fail(request):
        raise httpx2.ConnectError("Private host or credential details", request=request)

    with (
        CaptureClient(tmp_path, transport=httpx2.MockTransport(fail)) as client,
        pytest.raises(httpx2.ConnectError),
    ):
        client.get("https://private.example.test/")
    path, _ = records(tmp_path)
    report = summarize_trace(path)
    assert report["outcome"] == "error"
    assert report["error_type"] == "ConnectError"
    assert report["duration_seconds"] >= 0
    assert report["first_byte_seconds"] is None
    assert "Private host" not in path.read_text()


def test_non_stream_completion_and_usage(user_path, tmp_path, httpx2_mock, monkeypatch):
    from test_apisix import completion

    from llm.apisix_trace import summarize_trace

    create_profile(user_path)
    monkeypatch.setenv("LLM_APISIX_TRACE_DIR", str(tmp_path / "capture"))
    httpx2_mock.add_response(
        method="POST", url=BASE + "/chat/completions", json=completion("OK")
    )
    assert (
        ApisixChat("work", {"base_url": BASE, "key_alias": "apisix-work"}, MODEL)
        .prompt("Test", stream=False)
        .text()
        == "OK"
    )
    path, _ = records(tmp_path / "capture")
    report = summarize_trace(path)
    assert report["finish_reasons"] == ["stop"]
    assert report["usage"]["total_tokens"] == 2
    assert report["eof_observed"] is True
    assert report["done_received"] is False  # JSON completion has no SSE terminator.


def test_frames_bom_multiline_and_partial_record(tmp_path):
    from llm.apisix_trace import summarize_trace

    body = b'\xef\xbb\xbfdata: {"choices": [\rdata: {"delta":{"reasoning":"x"}}]}\r\rdata: invalid\n\ndata: [DONE]\n\n'
    rows = [
        {"type": "request", "elapsed_seconds": 0},
        {"type": "response", "elapsed_seconds": 1, "content_type": "text/event-stream"},
        {
            "type": "chunk",
            "elapsed_seconds": 301,
            "body_b64": base64.b64encode(body).decode(),
        },
    ]
    path = tmp_path / "capture.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + '\n{"type":"end",')
    report = summarize_trace(path)
    assert report["reasoning_events"] == 1
    assert report["non_json_events"] == 1
    assert report["done_received"] is True
    assert report["outcome"] == "unfinished"
    assert report["incomplete_capture_record"] is True
    assert report["duration_seconds"] is None
    assert report["observed_until_seconds"] == 301


def test_corrupt_compression_preserves_timing_report(tmp_path):
    from llm.apisix_trace import CaptureClient, summarize_trace

    def respond(request):
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream", "Content-Encoding": "gzip"},
            stream=IteratorStream([b"invalid gzip bytes"]),
        )

    with (
        CaptureClient(tmp_path, transport=httpx2.MockTransport(respond)) as client,
        pytest.raises(httpx2.DecodingError),
    ):
        client.get("https://private.example.test/")
    path, _ = records(tmp_path)
    report = summarize_trace(path)
    assert report["raw_body_bytes"] == len(b"invalid gzip bytes")
    assert report["decoding_error_type"] == "DecodingError"
    assert report["done_received"] is False


def test_sdk_reasoning_over_five_minutes_uses_observed_clock(
    user_path, tmp_path, httpx2_mock, monkeypatch
):
    from types import SimpleNamespace

    import llm.apisix_trace as capture_module
    from llm.apisix_trace import summarize_trace

    create_profile(user_path)
    monkeypatch.setenv("LLM_APISIX_TRACE_DIR", str(tmp_path / "capture"))
    now = [100.0]
    monkeypatch.setattr(
        capture_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    events = list(
        stream(
            [
                {"reasoning_content": "one"},
                {"reasoning_content": "two"},
                {"content": "answer"},
            ]
        )
    )

    class TimedStream(httpx2.SyncByteStream):
        def __iter__(self):
            for at, event in zip([102.0, 410.0, 416.0, 416.0, 416.0], events):
                now[0] = at
                yield event

    httpx2_mock.add_response(
        method="POST",
        url=BASE + "/chat/completions",
        stream=TimedStream(),
        headers={"Content-Type": "text/event-stream"},
    )
    assert (
        ApisixChat("work", {"base_url": BASE, "key_alias": "apisix-work"}, MODEL)
        .prompt("Synthetic clock test")
        .text()
        == "answer"
    )
    path, _ = records(tmp_path / "capture")
    report = summarize_trace(path)
    assert report["duration_seconds"] == 316.0
    assert report["reasoning_event_span_seconds"] == 308.0
    assert report["first_answer_seconds"] == 316.0
    assert report["done_received"] is True


def test_chat_capture_and_existing_duration_export(
    user_path, tmp_path, httpx2_mock, monkeypatch
):
    import re

    create_profile(user_path)
    config_path = user_path / "apisix.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["profiles"]["work"]["models"] = [MODEL]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("LLM_APISIX_TRACE_DIR", str(tmp_path / "capture"))
    httpx2_mock.add_response(
        method="POST",
        url=BASE + "/chat/completions",
        stream=IteratorStream(
            stream(
                [
                    {"reasoning_content": "Inspecting evidence"},
                    {"content": "END_INVESTIGATION_WORKBOOK"},
                ]
            )
        ),
        headers={"Content-Type": "text/event-stream"},
    )
    runner = CliRunner()
    chat = runner.invoke(
        cli, ["chat", "-m", "apisix/work/" + MODEL], input="Synthetic incident\nexit\n"
    )
    assert chat.exit_code == 0, chat.output
    assert "END_INVESTIGATION_WORKBOOK" in chat.output
    cid = re.search(r"Conversation ID: (\S+)", chat.output).group(1)
    logs = runner.invoke(cli, ["logs", "list", "--cid", cid, "--json", "-n", "0"])
    assert logs.exit_code == 0, logs.output
    turns = json.loads(logs.output)
    assert len(turns) == 1
    assert isinstance(turns[0]["duration_ms"], int)
    assert turns[0]["duration_ms"] >= 0
    path, _ = records(tmp_path / "capture")
    summary = runner.invoke(cli, ["apisix", "trace-summary", str(path)])
    assert summary.exit_code == 0, summary.output
    assert json.loads(summary.output)["reasoning_events"] == 1
