"""Shared foreground palette for terminal chat and provider status messages."""

import json
import math
import os
from decimal import Decimal

import click

PROGRAM_COLOR = (217, 119, 87)  # Warm orange, #d97757
USER_COLOR = (142, 199, 230)
LLM_COLOR = (230, 221, 214)
REASONING_COLOR = (174, 183, 196)
TOOL_COLOR = (196, 119, 158)  # Muted dark pink, #c4779e
RESULT_DISPLAY_NOTE = "Formatted display: JSON text expanded when valid; original payload/types in checkpoint."


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


def _json_display_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key would lose an original value")
        result[key] = value
    return result


def _json_display_float(text):
    value = float(text)
    if not math.isfinite(value) or Decimal(text) != Decimal(str(value)):
        raise ValueError("JSON number cannot be displayed without precision loss")
    return value


def _expand_json_text(value):
    if isinstance(value, str):
        original = value
        # Some tools encode a JSON document more than once. Unwrap only when
        # the final result is an object/array, never an ordinary/scalar string.
        while isinstance(value, str) and value.lstrip().startswith(("{", "[", '"')):
            try:
                value = json.loads(
                    value,
                    object_pairs_hook=_json_display_object,
                    parse_float=_json_display_float,
                    parse_constant=_json_display_float,
                )
            except (ValueError, RecursionError):
                return original
        if not isinstance(value, (dict, list)):
            return original
    if isinstance(value, dict):
        return {key: _expand_json_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_json_text(item) for item in value]
    return value


def tool_output_text(output) -> str:
    """Indent JSON, expanding encoded containers for display without mutation."""
    try:
        formatted = _expand_json_text(output)
    except RecursionError:
        # A deeply encoded string is still useful evidence in its original form.
        formatted = output
    return (
        formatted
        if isinstance(formatted, str)
        else json.dumps(formatted, ensure_ascii=False, indent=2)
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
    inherited = checkpoint.get("inherited_from")
    if inherited:
        notice += (
            f"\nInherited from conversation: {inherited['conversation_id']}"
            f"\nSource provider run: {inherited['provider_run_id']}"
            "\nPreserved evidence; not executed in this recovery request."
        )
    if preview:
        output = tool_output_text(checkpoint["output"])
        # Display bounds only: the model and saved output remain untouched.
        excerpt = "".join(output[:2000].splitlines(keepends=True)[:20])
        notice += "\n" + RESULT_DISPLAY_NOTE
        notice += "\nResult preview:\n" + (excerpt or "(empty string)")
        if len(excerpt) < len(output):
            notice += "\n[truncated; full result in HTML and checkpoint above]"
    return notice
