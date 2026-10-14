"""Terminal chat roles remain distinct without coloring data or backgrounds."""

import io
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import llm.cli
from llm.default_plugins.openwebui import _OpenWebUIStatusBar
from llm.parts import StreamEvent

ORANGE = "\x1b[38;2;217;119;87m"
BLUE = "\x1b[38;2;142;199;230m"
IVORY = "\x1b[38;2;230;221;214m"
GRAY = "\x1b[38;2;174;183;196m"


@pytest.mark.parametrize("no_color", [False, True])
def test_interactive_chat_colors_and_plain_journal(tmp_path, monkeypatch, no_color):
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "")
    monkeypatch.setenv("LLM_CHAT_EXPORT_DIR", str(tmp_path))
    monkeypatch.setattr(llm.cli, "_chat_turn_headers_enabled", lambda: True)
    prompts = iter(["hello", "quit"])
    monkeypatch.setattr(llm.cli, "_read_chat_prompt", lambda _: next(prompts))

    class Response:
        def stream_events(self):
            yield StreamEvent(type="reasoning", chunk="Think\nnext")
            yield StreamEvent(type="text", chunk="Answer")
            yield StreamEvent(type="reasoning", chunk="Check")
            yield StreamEvent(type="text", chunk="Done")

    @click.command()
    def app():
        llm.cli._run_chat(
            "test",
            lambda *args: Response(),
            conversation=SimpleNamespace(id="colors", name=None),
        )

    result = CliRunner().invoke(app, color=True, catch_exceptions=False)
    assert "── LLM ──" in click.unstyle(result.output)
    assert "Think\nnext\nAnswer\nCheck\nDone" in click.unstyle(result.output)
    if no_color:
        assert "\x1b[" not in result.output
    else:
        assert ORANGE + "Chatting with test" in result.stdout
        assert BLUE in result.stdout
        assert IVORY + "Answer\x1b[0m" in result.stdout
        assert GRAY + "Think\nnext\x1b[0m" in result.stderr
        assert "\x1b[2m" not in result.output
        assert "\x1b[48;" not in result.output
    journal = (tmp_path / "colors.jsonl").read_text("utf-8")
    assert "\\u001b" not in journal
    assert "Think\\nnext" in journal


@pytest.mark.parametrize("no_color", [False, True])
def test_status_bar_color_truncation_and_clear(monkeypatch, no_color):
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "")

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    monkeypatch.setattr("sys.stderr", terminal)
    monkeypatch.setattr("sys.stdout", terminal)
    bar = _OpenWebUIStatusBar()
    monkeypatch.setattr(bar, "_width", lambda: 40)
    bar.update("heartbeat during unfinished word", defer=True)
    assert terminal.getvalue() == ""
    assert bar.current == "heartbeat during unfinished word"
    bar.update("search " * 30)
    bar.clear()
    rendered = terminal.getvalue()
    line = rendered.split("\r")[2]
    assert len(click.unstyle(line)) == 39
    assert click.unstyle(line).startswith("Open WebUI │ ")
    assert click.unstyle(line).endswith("…")
    assert rendered.endswith("\r" + " " * 39 + "\r")
    assert (ORANGE in line) is not no_color
    if not no_color:
        assert line.endswith("\x1b[0m")


def test_tool_notices_are_persistent_pink_and_not_truncated(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    monkeypatch.setattr("sys.stderr", terminal)
    monkeypatch.setattr("sys.stdout", terminal)
    bar = _OpenWebUIStatusBar()
    monkeypatch.setattr(bar, "_width", lambda: 40)
    bar.update("waiting")
    call = "splunk_search\nArguments: " + "full query " * 20
    bar.update(call, kind="tool")
    before_clear = terminal.getvalue()
    assert call in before_clear
    assert "\x1b[38;2;196;119;158m" in before_clear
    assert before_clear.endswith("\x1b[0m\n")
    bar.clear()
    assert terminal.getvalue() == before_clear


@pytest.mark.parametrize("no_color", [False, True])
def test_composer_renders_user_input_and_orange_prompt(monkeypatch, no_color):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.output.vt100 import Vt100_Output

    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "")
    terminal = io.StringIO()
    output = Vt100_Output(
        terminal,
        lambda: Size(rows=24, columns=80),
        default_color_depth=ColorDepth.DEPTH_24_BIT,
    )
    with create_pipe_input() as pipe:
        monkeypatch.setattr(
            "prompt_toolkit.PromptSession",
            lambda **kwargs: PromptSession(input=pipe, output=output, **kwargs),
        )
        session = llm.cli._build_chat_prompt_session()
        pipe.send_text("my message\r")
        assert llm.cli._read_chat_prompt(session) == "my message"
    rendered = terminal.getvalue()
    assert ("38;2;217;119;87" in rendered) is not no_color
    assert ("38;2;142;199;230" in rendered) is not no_color
    assert "48;2" not in rendered


def test_redirected_chat_and_status_are_plain(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    prompts = iter(["hello", "quit"])
    monkeypatch.setattr(llm.cli, "_read_chat_prompt", lambda _: next(prompts))

    class Response:
        def stream_events(self):
            bar = _OpenWebUIStatusBar()
            bar.update("working")
            yield StreamEvent(type="reasoning", chunk="Thinking")
            yield StreamEvent(type="text", chunk="Answer")
            bar.clear()

    @click.command()
    def app():
        llm.cli._run_chat("test", lambda *args: Response())

    result = CliRunner().invoke(app, catch_exceptions=False)
    assert "[Open WebUI] working" in result.stderr
    assert "Thinking\nAnswer" in result.output
    assert "\x1b[" not in result.output
    assert "──" not in result.output
