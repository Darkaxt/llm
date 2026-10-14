import html
import json
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import llm.cli
from llm.chat_journal import append_record, ensure_session
from llm.parts import StreamEvent


def test_chat_html_updates_during_stream_and_preserves_visible_text(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    prompts = iter(["hello <team>\n  second line", "quit"])
    monkeypatch.setattr(llm.cli, "_read_chat_prompt", lambda _: next(prompts))
    report = tmp_path / "live.html"

    class Response:
        def stream_events(self):
            assert report.exists()
            assert "hello &lt;team&gt;\n  second line" in report.read_text("utf-8")
            append_record("live", {"type": "provider_status", "status": "working"})
            yield StreamEvent(type="reasoning", chunk="Think ")
            assert "Think " in report.read_text("utf-8")
            yield StreamEvent(type="reasoning", chunk="carefully\nnext")
            assert "Think carefully\nnext" in report.read_text("utf-8")
            yield StreamEvent(type="text", chunk="Answer <script>alert(1)</script>")
            assert "Answer &lt;script&gt;alert(1)&lt;/script&gt;" in report.read_text(
                "utf-8"
            )

    @click.command()
    def app():
        llm.cli._run_chat(
            "test",
            lambda *args: Response(),
            conversation=SimpleNamespace(id="live", name=None),
        )

    result = CliRunner().invoke(app, catch_exceptions=False)
    html = report.read_text("utf-8")
    assert "Autosave HTML:" in result.output
    assert "Type &#x27;exit&#x27; or &#x27;quit&#x27; to exit" in html
    assert 'class="user"' in html
    assert 'class="reasoning"' in html
    assert 'class="answer"' in html
    assert html.endswith("</html>\n")
    assert "<script>" not in html
    assert "#d97757" in html
    assert "#8ec7e6" in html
    assert "#e6ddd6" in html
    assert "#aeb7c4" in html


def test_journal_html_reopen_controls_and_current_status(tmp_path, monkeypatch):
    from llm import chat_html

    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("reopen", model="test")
    append_record("reopen", {"type": "user_message", "prompt": "original"})
    append_record(
        "reopen", {"type": "provider_request", "token": "hidden-secret", "messages": []}
    )
    append_record(
        "reopen",
        {
            "type": "assistant_stream",
            "event_type": "text",
            "chunk": "\x1b[31mfirst\x1b[0m",
        },
    )
    append_record("reopen", {"type": "provider_status", "status": "old-status"})
    append_record("reopen", {"type": "provider_status", "status": "current-status"})
    path = tmp_path / "reopen.html"
    html = path.read_text("utf-8")
    assert "old-status" not in html
    assert "current-status" in html
    assert "hidden-secret" not in html
    assert "\x1b" not in html
    # A fresh process rebuilds from the durable journal instead of duplicating
    # content or trusting a stale report after an interrupted write.
    chat_html._reports.clear()
    ensure_session("reopen", model="test")
    append_record(
        "reopen", {"type": "assistant_stream", "event_type": "text", "chunk": " second"}
    )
    html = path.read_text("utf-8")
    assert "first second" in html
    assert html.count("original") == 1
    assert html.count("first second") == 1


def test_html_io_failure_does_not_break_durable_journal(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    # The HTML path is occupied by an incompatible filesystem object.
    (tmp_path / "blocked.html").mkdir()
    ensure_session("blocked", model="test")
    append_record("blocked", {"type": "user_message", "prompt": "saved"}, durable=True)
    records = [
        json.loads(line)
        for line in (tmp_path / "blocked.jsonl").read_text("utf-8").splitlines()
    ]
    assert records[-1]["prompt"] == "saved"
    assert capsys.readouterr().err.count("HTML report unavailable:") == 1


def test_failed_rebuild_preserves_previous_report(tmp_path, monkeypatch, capsys):
    from llm import chat_html

    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("preserve", model="test")
    append_record(
        "preserve",
        {"type": "assistant_stream", "event_type": "text", "chunk": "saved reply"},
    )
    report = tmp_path / "preserve.html"
    previous = report.read_bytes()
    chat_html._reports.clear()
    monkeypatch.setattr(
        chat_html._HTMLReport,
        "append",
        lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    ensure_session("preserve", model="test")
    assert report.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
    append_record(
        "preserve",
        {"type": "user_message", "prompt": "journal still works"},
        durable=True,
    )
    assert "journal still works" in (tmp_path / "preserve.jsonl").read_text("utf-8")
    assert capsys.readouterr().err.count("HTML report unavailable:") == 1


def test_hidden_reasoning_cancellation_and_error_are_rendered(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("states", model="test")
    append_record("states", {"type": "display_options", "show_reasoning": False})
    append_record(
        "states",
        {
            "type": "assistant_stream",
            "event_type": "reasoning",
            "chunk": "hidden thought",
        },
    )
    append_record(
        "states",
        {"type": "assistant_stream", "event_type": "text", "chunk": "partial answer"},
    )
    append_record("states", {"type": "turn_cancelled"})
    append_record("states", {"type": "turn_error", "error": "bad <request>"})
    html = (tmp_path / "states.html").read_text("utf-8")
    assert "hidden thought" not in html
    assert "partial answer" in html
    assert "Cancelled current response." in html
    assert "Model error: bad &lt;request&gt;" in html
    assert html.endswith("</html>\n")


def test_html_stream_updates_do_not_read_or_rewrite_history(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("incremental", model="test")
    append_record(
        "incremental",
        {"type": "assistant_stream", "event_type": "text", "chunk": "prefix " * 10000},
    )
    report = tmp_path / "incremental.html"
    real_open = Path.open
    writes = []

    class TrackedFile:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, data):
            writes.append(len(data))
            return self.handle.write(data)

        def read(self, *args):
            raise AssertionError("Incremental HTML update must not reread history")

    def tracked_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        return TrackedFile(handle) if path == report else handle

    monkeypatch.setattr(Path, "open", tracked_open)
    append_record(
        "incremental",
        {"type": "assistant_stream", "event_type": "text", "chunk": "suffix"},
    )
    assert writes
    assert sum(writes) < 2000


@pytest.mark.parametrize(
    "tool_output",
    [
        "timestamp: 2026-10-14T12:00:00Z",
        "begin " + "large result " * 300 + "END_OF_RESULT",
        "\n".join(f"result line {i}" for i in range(40)),
        {
            "content": [{"type": "text", "text": "<script>unsafe</script>"}],
            "extra": [1, None],
        },
        "",
        None,
    ],
    ids=["timestamp", "long", "many-lines", "structured", "empty", "null"],
)
def test_provider_tools_are_inline_auditable_and_in_html(
    tmp_path, monkeypatch, tool_output
):
    from llm.default_plugins import openwebui

    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {
            "url": "https://example.test",
            "token": "jwt",
            "enabled_tool_ids": ["server:mcp:splunk-mcp"],
        },
    )
    monkeypatch.setattr(
        openwebui,
        "_client",
        lambda _: SimpleNamespace(
            base_url="https://example.test", token="jwt", timeout=1200
        ),
    )
    monkeypatch.setattr(
        openwebui,
        "_get_model_item",
        lambda *args: {
            "id": "glm-5.3",
            "info": {"meta": {"capabilities": {"builtin_tools": True}}},
        },
    )
    monkeypatch.setattr(openwebui, "_server_config", lambda _: {"version": "0.12.0"})
    monkeypatch.setattr(openwebui, "_enabled_knowledge_items", lambda *args: [])
    prompts = iter(["synthetic validation", "quit"])
    monkeypatch.setattr(llm.cli, "_read_chat_prompt", lambda _: next(prompts))
    conversation = openwebui.OpenWebUIModel("glm-5.3").conversation()

    async def runner(**kwargs):
        kwargs["on_reasoning"]("Check the source")
        kwargs["on_tool"]("↳ splunk_search ...")
        kwargs["on_checkpoint"](
            {
                "call_id": "call-1",
                "name": "splunk_search",
                "arguments": {
                    "query": "index=synthetic | stats count",
                    "earliest": "-1h",
                },
                "output": tool_output,
            }
        )
        kwargs["on_status"]("waiting for model continuation")
        kwargs["on_text"]("Final answer")
        return {
            "answer": "Final answer",
            "reasoning": "Check the source",
            "native_checkpoint_count": 1,
        }

    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", runner)

    @click.command()
    def app():
        llm.cli._run_chat(
            conversation.model.model_id,
            lambda prompt, *args: conversation.prompt(prompt),
            conversation=conversation,
        )

    result = CliRunner().invoke(app, catch_exceptions=False)
    assert "index=synthetic | stats count" in result.stderr
    assert "Result checkpoint:" in result.stderr
    assert "Result preview:" in result.stderr
    result_text = (
        tool_output
        if isinstance(tool_output, str)
        else json.dumps(tool_output, ensure_ascii=False, indent=2)
    )
    if len(result_text) > 2000 or len(result_text.splitlines()) > 20:
        preview = result.stderr.split("Result preview:\n", 1)[1].split(
            "\n[truncated", 1
        )[0]
        assert len(preview) <= 2000
        assert len(preview.splitlines()) <= 20
        assert "[truncated; full result in HTML and checkpoint above]" in result.stderr
        assert "END_OF_RESULT" not in result.stderr
        assert "result line 39" not in result.stderr
    else:
        assert result_text in result.stderr
        assert "[truncated" not in result.stderr
    assert (
        result.output.index("Check the source")
        < result.output.index("Arguments:")
        < result.output.index("Final answer")
    )
    report = (tmp_path / f"{conversation.id}.html").read_text("utf-8")
    assert 'class="tool"' in report
    assert "index=synthetic | stats count" in report
    assert "Result checkpoint:" in report
    assert html.escape(result_text, quote=True) in report
    assert "[truncated" not in report
    assert "<script>" not in report
    assert '<details class="tool-result">' in report
    assert "<summary>Full result · splunk_search</summary>" in report

    class DetailsParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.panels = []

        def handle_starttag(self, tag, attrs):
            if tag == "details":
                self.panels.append(dict(attrs))

    parser = DetailsParser()
    parser.feed(report)
    assert parser.panels == [{"class": "tool-metadata"}, {"class": "tool-result"}]
    assert "<summary>Call metadata</summary>" in report
    assert "#c4779e" in report
    assert (
        report.index("Check the source")
        < report.index("Arguments:")
        < report.index("Final answer")
    )
    from llm import chat_html

    chat_html._reports.clear()
    ensure_session(conversation.id, model=conversation.model.model_id)
    rebuilt = (tmp_path / f"{conversation.id}.html").read_text("utf-8")
    assert rebuilt.count('<details class="tool-result">') == 1
    assert html.escape(result_text, quote=True) in rebuilt
    # A repeated display record for the same returned call cannot add a second
    # result panel when the report is regenerated.
    records = [
        json.loads(line)
        for line in (tmp_path / f"{conversation.id}.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    pointer = next(row for row in records if row.get("type") == "tool_result_display")
    append_record(conversation.id, pointer)
    assert (tmp_path / f"{conversation.id}.html").read_text("utf-8").count(
        '<details class="tool-result">'
    ) == 1


def test_corrupt_result_does_not_break_journal_or_replace_readable_report(
    tmp_path, monkeypatch, capsys
):
    from llm.openwebui_recovery import save_tool_checkpoint

    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("corrupt-result", model="test")
    pointer = save_tool_checkpoint(
        "corrupt-result",
        "run",
        {"call_id": "call", "name": "tool", "arguments": {}, "output": "real result"},
    )
    report = tmp_path / "corrupt-result.html"
    previous = report.read_bytes()
    (tmp_path / pointer["checkpoint_file"]).write_text("tampered", encoding="utf-8")
    append_record("corrupt-result", {**pointer, "type": "tool_result_display"})
    append_record(
        "corrupt-result",
        {"type": "assistant_stream", "event_type": "text", "chunk": "continues"},
        durable=True,
    )
    assert report.read_bytes() == previous
    assert "tampered" not in report.read_text("utf-8")
    assert "continues" in (tmp_path / "corrupt-result.jsonl").read_text("utf-8")
    assert "HTML report unavailable:" in capsys.readouterr().err


def test_existing_completed_call_notice_rebuilds_with_full_result(
    tmp_path, monkeypatch
):
    from llm import chat_html
    from llm.chat_ui import checkpoint_notice
    from llm.openwebui_recovery import save_tool_checkpoint

    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    ensure_session("old-notice", model="test")
    call = {
        "call_id": "call",
        "name": "get_current_timestamp",
        "arguments": {},
        "output": {"timestamp": "2026-10-14T12:39:07Z"},
    }
    # The worker can save a checkpoint before the UI has consumed reasoning.
    pointer = save_tool_checkpoint("old-notice", "run", call)
    notice = checkpoint_notice(call, pointer, preview=False)
    append_record(
        "old-notice", {"type": "provider_tool", "tool": notice, "display": False}
    )
    append_record(
        "old-notice",
        {"type": "assistant_stream", "event_type": "reasoning", "chunk": "Before call"},
    )
    append_record("old-notice", {"type": "tool_display", "tool": notice})
    append_record(
        "old-notice",
        {"type": "assistant_stream", "event_type": "text", "chunk": "After call"},
    )
    chat_html._reports.clear()
    ensure_session("old-notice", model="test")
    report = (tmp_path / "old-notice.html").read_text("utf-8")
    assert report.count('<details class="tool-result">') == 1
    assert (
        report.index("Before call")
        < report.index("2026-10-14T12:39:07Z")
        < report.index("After call")
    )
    assert "<summary>Call metadata</summary>" in report


def test_tty_status_events_do_not_split_streamed_words(tmp_path, monkeypatch):
    from llm.default_plugins import openwebui

    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        openwebui,
        "_load_config",
        lambda: {"enabled_tool_ids": ["server:mcp:splunk-mcp"]},
    )
    monkeypatch.setattr(
        openwebui,
        "_client",
        lambda _: SimpleNamespace(
            base_url="https://example.test", token="jwt", timeout=1200
        ),
    )
    monkeypatch.setattr(openwebui, "_get_model_item", lambda *args: {"id": "glm-5.3"})
    monkeypatch.setattr(openwebui, "_server_config", lambda _: {"version": "0.12.0"})
    monkeypatch.setattr(openwebui, "_enabled_knowledge_items", lambda *args: [])
    original_init = openwebui._OpenWebUIStatusBar.__init__

    def tty_init(self):
        original_init(self)
        self.enabled = True

    monkeypatch.setattr(openwebui._OpenWebUIStatusBar, "__init__", tty_init)

    async def runner(**kwargs):
        for kind, chunks in (
            ("reasoning", ["Current ", "tim", "e 2026", "-10-14."]),
            ("text", ["Final ", "ans", "wer."]),
        ):
            for chunk in chunks:
                kwargs["on_" + kind](chunk)
                kwargs["on_status"]("waiting for model (15s phase)")
        return {"answer": "Final answer.", "reasoning": "Current time 2026-10-14."}

    monkeypatch.setattr(openwebui, "run_chat_with_tools_with_files", runner)
    conversation = openwebui.OpenWebUIModel("glm-5.3").conversation()

    @click.command()
    def app():
        llm.cli.display_stream_events(
            conversation.prompt("synthetic").stream_events(), chat=True
        )

    result = CliRunner().invoke(app, catch_exceptions=False)
    assert "Current time 2026-10-14." in click.unstyle(result.output)
    assert "Final answer." in click.unstyle(result.output)
