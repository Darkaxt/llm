import json
import re
import sys
import textwrap
from unittest.mock import ANY

import pytest
from click.testing import CliRunner

import llm.cli
from llm.logs import LogStore, merged_log_rows


def _strip_chat_session_banner(output):
    return re.sub(
        r"Conversation ID: [^\\n]+\\nAutosave JSONL: [^\\n]+\\n",
        "",
        output,
    )


def logged_rows(db):
    """Chronological log rows from the store, reduced to the fields
    these tests care about."""
    rows = merged_log_rows(LogStore(db))
    rows.reverse()
    return [
        {
            "model": row["model"],
            "prompt": row["prompt"],
            "system": row["system"],
            "options_json": row["options_json"],
            "response": row["response"],
            "conversation_id": row["conversation_id"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        }
        for row in rows
    ]


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_basic(mock_model, logs_db):
    runner = CliRunner()
    mock_model.enqueue(["one world"])
    mock_model.enqueue(["one again"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock"],
        input="Hi\nHi two\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert _strip_chat_session_banner(result.output) == (
        "Chatting with mock"
        "\nType 'exit' or 'quit' to exit"
        "\nEnter sends; Shift+Enter adds a newline (Ctrl+J / Alt+Enter fallback)"
        "\nType '!edit' to open your default editor and modify the prompt"
        "\nPress Ctrl+C during generation to cancel the current response"
        "\nType '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments"
        "\n> Hi"
        "\none world"
        "\n> Hi two"
        "\none again"
        "\n> quit"
        "\n"
    )
    # Should have logged
    threads = list(logs_db["threads"].rows)
    assert threads[0]["name"] == "Hi"
    conversation_id = threads[0]["id"]
    responses = logged_rows(logs_db)
    assert responses == [
        {
            "model": "mock",
            "prompt": "Hi",
            "system": None,
            "options_json": "{}",
            "response": "one world",
            "conversation_id": conversation_id,
            "input_tokens": 1,
            "output_tokens": 1,
        },
        {
            "model": "mock",
            "prompt": "Hi two",
            "system": None,
            "options_json": "{}",
            "response": "one again",
            "conversation_id": conversation_id,
            "input_tokens": 2,
            "output_tokens": 1,
        },
    ]
    # Now continue that conversation
    mock_model.enqueue(["continued"])
    result2 = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock", "-c"],
        input="Continue\nquit\n",
        catch_exceptions=False,
    )
    assert result2.exit_code == 0
    assert _strip_chat_session_banner(result2.output) == (
        "Chatting with mock"
        "\nType 'exit' or 'quit' to exit"
        "\nEnter sends; Shift+Enter adds a newline (Ctrl+J / Alt+Enter fallback)"
        "\nType '!edit' to open your default editor and modify the prompt"
        "\nPress Ctrl+C during generation to cancel the current response"
        "\nType '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments"
        "\n> Continue"
        "\ncontinued"
        "\n> quit"
        "\n"
    )
    new_responses = logged_rows(logs_db)[len(responses) :]
    assert new_responses == [
        {
            "model": "mock",
            "prompt": "Continue",
            "system": None,
            "options_json": "{}",
            "response": "continued",
            "conversation_id": conversation_id,
            "input_tokens": 1,
            "output_tokens": 1,
        }
    ]


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_system(mock_model, logs_db):
    runner = CliRunner()
    mock_model.enqueue(["I am mean"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock", "--system", "You are mean"],
        input="Hi\nquit\n",
    )
    assert result.exit_code == 0
    assert _strip_chat_session_banner(result.output) == (
        "Chatting with mock"
        "\nType 'exit' or 'quit' to exit"
        "\nEnter sends; Shift+Enter adds a newline (Ctrl+J / Alt+Enter fallback)"
        "\nType '!edit' to open your default editor and modify the prompt"
        "\nPress Ctrl+C during generation to cancel the current response"
        "\nType '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments"
        "\n> Hi"
        "\nI am mean"
        "\n> quit"
        "\n"
    )
    responses = logged_rows(logs_db)
    assert responses == [
        {
            "model": "mock",
            "prompt": "Hi",
            "system": "You are mean",
            "options_json": "{}",
            "response": "I am mean",
            "conversation_id": ANY,
            "input_tokens": 1,
            "output_tokens": 1,
        }
    ]


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_options(mock_model, logs_db, user_path):
    options_path = user_path / "model_options.json"
    options_path.write_text(json.dumps({"mock": {"max_tokens": "5"}}), "utf-8")

    runner = CliRunner()
    mock_model.enqueue(["Default options response"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock"],
        input="Hi\nquit\n",
    )
    assert result.exit_code == 0
    mock_model.enqueue(["Override options response"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock", "--option", "max_tokens", "10"],
        input="Hi with override\nquit\n",
    )
    assert result.exit_code == 0
    responses = logged_rows(logs_db)
    assert responses == [
        {
            "model": "mock",
            "prompt": "Hi",
            "system": None,
            "options_json": '{"max_tokens": 5}',
            "response": "Default options response",
            "conversation_id": ANY,
            "input_tokens": 1,
            "output_tokens": 1,
        },
        {
            "model": "mock",
            "prompt": "Hi with override",
            "system": None,
            "options_json": '{"max_tokens": 10}',
            "response": "Override options response",
            "conversation_id": ANY,
            "input_tokens": 3,
            "output_tokens": 1,
        },
    ]


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
@pytest.mark.parametrize(
    "input,expected",
    (
        (
            "Hi\n!multi\nthis is multiple lines\nuntil the !end\n!end\nquit\n",
            [
                {"prompt": "Hi", "response": "One\n"},
                {
                    "prompt": "this is multiple lines\nuntil the !end",
                    "response": "Two\n",
                },
            ],
        ),
        # quit should not work within !multi
        (
            "!multi\nthis is multiple lines\nquit\nuntil the !end\n!end\nquit\n",
            [
                {
                    "prompt": "this is multiple lines\nquit\nuntil the !end",
                    "response": "One\n",
                }
            ],
        ),
        # Try custom delimiter
        (
            "!multi abc\nCustom delimiter\n!end\n!end 123\n!end abc\nquit\n",
            [{"prompt": "Custom delimiter\n!end\n!end 123", "response": "One\n"}],
        ),
    ),
)
def test_chat_multi(mock_model, logs_db, input, expected):
    runner = CliRunner()
    mock_model.enqueue(["One\n"])
    mock_model.enqueue(["Two\n"])
    mock_model.enqueue(["Three\n"])
    result = runner.invoke(
        llm.cli.cli, ["chat", "-m", "mock", "--option", "max_tokens", "10"], input=input
    )
    assert result.exit_code == 0
    rows = [
        {"prompt": row["prompt"], "response": row["response"]}
        for row in logged_rows(logs_db)
    ]
    assert rows == expected


@pytest.mark.parametrize("custom_database_path", (False, True))
def test_llm_chat_creates_log_database(
    db_factory, tmpdir, monkeypatch, custom_database_path
):
    user_path = tmpdir / "user"
    custom_db_path = tmpdir / "custom_log.db"
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    runner = CliRunner()
    args = ["chat", "-m", "mock"]
    if custom_database_path:
        args.extend(["--database", str(custom_db_path)])
    result = runner.invoke(
        llm.cli.cli,
        args,
        catch_exceptions=False,
        input="Hi\nHi two\nquit\n",
    )
    assert result.exit_code == 0
    # Should have created user_path and put a logs.db in it
    if custom_database_path:
        assert custom_db_path.exists()
        db_path = str(custom_db_path)
    else:
        assert (user_path / "logs.db").exists()
        db_path = str(user_path / "logs.db")
    assert db_factory(db_path)["turns"].count == 2


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_tools(logs_db):
    runner = CliRunner()
    functions = textwrap.dedent("""
    def upper(text: str) -> str:
        "Convert text to upper case"
        return text.upper()                         
    """)
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "echo", "--functions", functions],
        input="\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Convert hello to uppercase",
                        "tool_calls": [
                            {"name": "upper", "arguments": {"text": "hello"}}
                        ],
                    }
                ),
                "quit",
            ]
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    normalized_output = re.sub(r"tc_[0-9a-z]{26}", "tc_TCID", _strip_chat_session_banner(result.output))
    assert normalized_output == (
        "Chatting with echo\n"
        "Type 'exit' or 'quit' to exit\n"
        "Enter sends; Shift+Enter adds a newline (Ctrl+J / Alt+Enter fallback)\n"
        "Type '!edit' to open your default editor and modify the prompt\n"
        "Press Ctrl+C during generation to cancel the current response\n"
        "Type '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments\n"
        '> {"prompt": "Convert hello to uppercase", "tool_calls": [{"name": "upper", '
        '"arguments": {"text": "hello"}}]}\n'
        "{\n"
        '  "prompt": "Convert hello to uppercase",\n'
        '  "system": "",\n'
        '  "attachments": [],\n'
        '  "stream": true,\n'
        '  "previous": []\n'
        "} {\n"
        '  "prompt": "",\n'
        '  "system": "",\n'
        '  "attachments": [],\n'
        '  "stream": true,\n'
        '  "previous": [\n'
        "    {\n"
        '      "prompt": "{\\"prompt\\": \\"Convert hello to uppercase\\", '
        '\\"tool_calls\\": [{\\"name\\": \\"upper\\", \\"arguments\\": {\\"text\\": '
        '\\"hello\\"}}]}"\n'
        "    }\n"
        "  ],\n"
        '  "tool_results": [\n'
        "    {\n"
        '      "name": "upper",\n'
        '      "output": "HELLO",\n'
        '      "tool_call_id": "tc_TCID"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "> quit\n"
    )


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_fragments(tmpdir):
    path1 = str(tmpdir / "frag1.txt")
    path2 = str(tmpdir / "frag2.txt")
    with open(path1, "w") as fp:
        fp.write("one")
    with open(path2, "w") as fp:
        fp.write("two")
    runner = CliRunner()
    output = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "echo", "-f", path1],
        input=(f"hi\n!fragment {path2}\nquit\n"),
    ).output
    assert '"prompt": "one' in output
    assert '"prompt": "two"' in output



def test_run_chat_ctrl_c_cancels_current_response(monkeypatch, capsys):
    prompts = iter(["hello", "quit"])
    monkeypatch.setattr(
        llm.cli.click,
        "prompt",
        lambda *args, **kwargs: next(prompts),
    )

    class FakeResponse:
        def stream_events(self):
            return iter(())

    monkeypatch.setattr(
        llm.cli,
        "display_stream_events",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    after = []
    llm.cli._run_chat(
        "mock",
        lambda prompt, fragments, attachments: FakeResponse(),
        after_response=lambda response: after.append(response),
    )

    captured = capsys.readouterr()
    assert "Cancelled current response." in captured.err
    assert after == []



@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_autosaves_jsonl(mock_model, logs_db, user_path):
    runner = CliRunner()
    mock_model.enqueue(["saved answer"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock"],
        input="saved prompt\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    match = re.search(r"Conversation ID: ([^\n]+)", result.output)
    assert match is not None
    conversation_id = match.group(1).strip()

    export_path = user_path / "conversations" / f"{conversation_id}.jsonl"
    assert export_path.exists()

    records = [
        json.loads(line)
        for line in export_path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["type"] == "conversation"
    assert records[0]["conversation_id"] == conversation_id
    assert records[0]["model"] == "mock"

    turns = [record for record in records if record["type"] == "turn"]
    assert len(turns) == 1
    assert turns[0]["conversation_id"] == conversation_id
    assert turns[0]["prompt"] == "saved prompt"
    assert turns[0]["response"] == "saved answer"



def test_read_chat_prompt_noninteractive_uses_click(monkeypatch):
    monkeypatch.setattr(
        llm.cli.click,
        "prompt",
        lambda *args, **kwargs: "hello",
    )
    assert llm.cli._read_chat_prompt(None) == "hello"



def test_chat_prompt_session_shift_enter_sequences():
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    llm.cli._build_chat_prompt_session()

    assert ANSI_SEQUENCES["\x1b[27;2;13~"] == Keys.F24
    assert ANSI_SEQUENCES["\x1b[13;2u"] == Keys.F24
