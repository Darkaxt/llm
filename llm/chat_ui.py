"""Shared foreground palette for terminal chat and provider status messages."""

import json
import os

import click

PROGRAM_COLOR = (217, 119, 87)  # Warm orange, #d97757
USER_COLOR = (142, 199, 230)
LLM_COLOR = (230, 221, 214)
REASONING_COLOR = (174, 183, 196)
TOOL_COLOR = (196, 119, 158)  # Muted dark pink, #c4779e


def colors_enabled() -> bool:
    context = click.get_current_context(silent=True)
    return "NO_COLOR" not in os.environ and (
        context is None or context.color is not False
    )


def chat_style(text: str, *, fg=PROGRAM_COLOR, bold=None) -> str:
    if not colors_enabled():
        return text
    # Foreground only: retain the terminal's background and reset after each
    # fragment so streamed text cannot color the next status or input line.
    return click.style(text, fg=fg, bold=bold)


def echo_program(text: str, **kwargs) -> None:
    click.echo(chat_style(text), **kwargs)


def cancellation_notice(record: dict) -> str:
    message = "Cancelled current response."
    if record.get("source") == "process_signal":
        message += f"\nCancellation source: process signal {record.get('signal')}."
    elif record.get("source"):
        message += (
            "\nCancellation source: local KeyboardInterrupt (origin not identified)."
        )
    return message


def tool_output_text(output) -> str:
    """Keep string results intact and all structured result fields visible."""
    return (
        output
        if isinstance(output, str)
        else json.dumps(output, ensure_ascii=False, indent=2)
    )


def checkpoint_notice(checkpoint: dict, pointer: dict, *, preview: bool = True) -> str:
    """Render actual call arguments and its durably saved result reference."""
    arguments = checkpoint["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, indent=2)
    notice = (
        f"↳ {checkpoint['name']} · returned result\n"
        f"Call ID: {checkpoint['call_id']}\nArguments:\n{arguments}\n"
        f"Result checkpoint: {pointer['checkpoint_file']}\n"
        f"Checkpoint: {pointer['size_bytes']} bytes · SHA-256 {pointer['sha256']}"
    )
    if preview:
        output = tool_output_text(checkpoint["output"])
        # Display bounds only: the model and saved output remain untouched.
        excerpt = "".join(output[:2000].splitlines(keepends=True)[:20])
        notice += "\nResult preview:\n" + (excerpt or "(empty string)")
        if len(excerpt) < len(output):
            notice += "\n[truncated; full result in HTML and checkpoint above]"
    return notice
