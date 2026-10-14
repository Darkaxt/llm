import json

import pytest

from llm import chat_html
from llm.chat_journal import append_record, ensure_session


def test_live_timing_preserves_lines_and_exposes_exact_event_gaps(
    tmp_path, monkeypatch, html_transcript
):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    base = 1791983182.125
    ensure_session("timed", model="synthetic")
    events = [
        {
            "type": "user_message",
            "prompt": "Investigate\n  carefully",
            "timestamp": base,
        },
        {
            "type": "assistant_stream",
            "event_type": "reasoning",
            "chunk": "Th",
            "timestamp": base + 1,
        },
        {"type": "provider_status", "status": "still working", "timestamp": base + 16},
        {
            "type": "assistant_stream",
            "event_type": "reasoning",
            "chunk": "ink\nnext",
            "timestamp": base + 460,
        },
        {
            "type": "assistant_stream",
            "event_type": "text",
            "chunk": "Final <answer>",
            "timestamp": base + 461,
        },
        {"type": "turn_completed", "timestamp": base + 462},
    ]
    for record in events:
        append_record("timed", record)
        report = (tmp_path / "timed.html").read_text("utf-8")
        assert report.endswith("</html>\n")
    assert 'id="event-times"' in report
    assert "Individual events" in report
    assert "2026-10-14 13:06:22.125 UTC" in report
    assert "Δ 444.000s" in report
    assert "Δ 15.000s" in report
    assert "still working" in report
    assert html_transcript(report).split("\n") == [
        "Investigate",
        "  carefully",
        "Think",
        "next",
        "Final <answer>",
        "Response completed",
    ]
    assert "<answer>" not in report
    journal = tmp_path / "timed.jsonl"
    original = journal.read_bytes()
    chat_html._reports.clear()
    chat_html.ensure_html_report(journal)
    assert (tmp_path / "timed.html").read_text("utf-8") == report
    assert journal.read_bytes() == original


@pytest.mark.parametrize(
    "timestamp", [None, "bad", True, float("nan"), float("inf"), 1e100]
)
def test_missing_or_invalid_timestamp_stays_explicit(tmp_path, monkeypatch, timestamp):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("unknown-time", model="synthetic")
    append_record(
        "unknown-time",
        {"type": "program_message", "text": "still visible", "timestamp": timestamp},
    )
    report = (tmp_path / "unknown-time.html").read_text("utf-8")
    assert "Time unavailable" in report
    assert "Δ unavailable" in report
    assert "still visible" in report
    assert (tmp_path / "unknown-time.jsonl").is_file()


def test_clock_regression_hidden_events_and_terminal_states(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("clock", model="synthetic")
    for record in [
        {"type": "display_options", "show_reasoning": False},
        {"type": "user_message", "prompt": "start", "timestamp": 100},
        {
            "type": "assistant_stream",
            "event_type": "reasoning",
            "chunk": "SECRET",
            "timestamp": 1000,
        },
        {
            "type": "provider_status",
            "status": "DUPLICATE",
            "display": False,
            "timestamp": 1001,
        },
        {"type": "provider_request", "token": "TOKEN", "timestamp": 1002},
        {
            "type": "assistant_stream",
            "event_type": "text",
            "chunk": "reply",
            "timestamp": 110,
        },
        {
            "type": "turn_cancelled",
            "source": "unknown_local_interrupt",
            "timestamp": 109,
        },
        {"type": "turn_error", "error": "bad <error>", "timestamp": 111},
    ]:
        append_record("clock", record)
    report = (tmp_path / "clock.html").read_text("utf-8")
    assert "Δ 10.000s" in report
    assert "clock moved backwards" in report
    assert "Δ -" not in report
    assert all(secret not in report for secret in ("SECRET", "DUPLICATE", "TOKEN"))
    assert "Cancelled current response" in report
    assert "bad &lt;error&gt;" in report
    records = [
        json.loads(line)
        for line in (tmp_path / "clock.jsonl").read_text("utf-8").splitlines()
    ]
    assert records[-1]["error"] == "bad <error>"


def test_fragmented_newlines_empty_chunks_and_recovery_completion(
    tmp_path, monkeypatch, html_transcript
):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("newlines", model="synthetic")
    for index, chunk in enumerate(["  fir", "st\n", "\n", "", "\tsecond", "\nlast"]):
        append_record(
            "newlines",
            {
                "type": "assistant_stream",
                "event_type": "text",
                "chunk": chunk,
                "timestamp": 100 + index,
            },
        )
    append_record("newlines", {"type": "recovery_completed", "timestamp": 110})
    report = (tmp_path / "newlines.html").read_text("utf-8")
    assert html_transcript(report).split("\n") == [
        "  first",
        "",
        "\tsecond",
        "last",
        "Response completed",
    ]
    assert "Empty stream event" in report
    assert "Δ 5.000s" in report
