"""Shared foreground palette for terminal chat and provider status messages."""

import os

import click

PROGRAM_COLOR = (217, 119, 87)  # Warm orange, #d97757
USER_COLOR = (142, 199, 230)
LLM_COLOR = (230, 221, 214)
REASONING_COLOR = (174, 183, 196)


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
