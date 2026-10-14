"""Incremental, offline HTML companion to the authoritative chat journal."""

from __future__ import annotations

import html
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chat_ui import (
    LLM_COLOR,
    PROGRAM_COLOR,
    REASONING_COLOR,
    TOOL_COLOR,
    USER_COLOR,
    cancellation_notice,
    checkpoint_notice,
    echo_program,
    tool_output_text,
)

_reports: dict[Path, _HTMLReport] = {}
_disabled: set[Path] = set()
_ansi = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
_controls = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _text(value: Any) -> str:
    return html.escape(_controls.sub("", _ansi.sub("", str(value))), quote=True)


def report_path(journal: Path) -> Path:
    return journal.with_suffix(".html")


class _HTMLReport:
    def __init__(self, journal: Path, *, path: Path):
        self.path = path
        self.conversation_id = journal.stem
        self.tool_results: set[tuple[str, str]] = set()
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.open_kind: str | None = None
        self.turn_id: str | None = None
        self.show_reasoning = True
        self.status = "Conversation started"
        self.model = ""
        self.updated = ""
        self.line_open = False
        self.previous_timestamp: float | None = None
        self.status_timing = ""
        colors = {
            key: "#{:02x}{:02x}{:02x}".format(*value)
            for key, value in (
                ("program", PROGRAM_COLOR),
                ("user", USER_COLOR),
                ("answer", LLM_COLOR),
                ("reasoning", REASONING_COLOR),
                ("tool", TOOL_COLOR),
            )
        }
        palette = "".join(f"--{key}:{value};" for key, value in colors.items())
        header = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Conversation {_text(journal.stem)}</title>
<style>
:root{{{palette}color-scheme:dark;background:#1f1f1f;color:var(--answer)}}
*{{box-sizing:border-box}}body{{margin:0;font:16px/1.6 Consolas,'Liberation Mono',monospace}}
header,main{{max-width:1100px;margin:auto;padding:24px}}header{{border-bottom:1px solid #45413e}}
h1{{font-size:20px;margin:0 0 8px;color:var(--program)}}header p{{margin:0;overflow-wrap:anywhere}}
main{{padding-bottom:10rem}}section{{margin:0 0 24px;border-left:2px solid currentColor;padding-left:16px}}
h2{{font-size:14px;line-height:1.4;margin:0 0 8px}}pre{{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}}
.timed-lines{{white-space:normal}}
.transcript-line{{display:grid;grid-template-columns:16rem minmax(0,1fr);gap:12px}}
.line-content,.event-content{{white-space:pre-wrap;overflow-wrap:anywhere;min-width:0}}
.line-time,.event-time{{font-size:11px;line-height:1.5;color:var(--reasoning);user-select:none;white-space:normal;overflow-wrap:anywhere}}
.event-gap{{display:block}}.event-time,.status-event{{display:none}}
.timing-help{{max-width:1100px;margin:0 auto 16px;padding:0 24px;font-size:12px;color:var(--reasoning)}}
#event-times{{margin-left:max(24px,calc((100% - 1052px)/2));accent-color:var(--program)}}
label[for=event-times]{{cursor:pointer;color:var(--program);font-size:13px}}
#event-times:focus-visible{{outline:2px solid var(--program);outline-offset:3px}}
#event-times:checked ~ main .stream-lines .line-time{{display:none}}
#event-times:checked ~ main .stream-event,#event-times:checked ~ main .status-event{{display:grid;grid-template-columns:16rem minmax(0,1fr);gap:12px}}
#event-times:checked ~ main .event-time{{display:block}}
#event-times:checked ~ main .stream-lines .transcript-line{{display:block}}
#event-times:checked ~ main .status-event{{color:var(--program);font-size:13px}}
.program{{color:var(--program);margin:0 0 12px;white-space:normal;overflow-wrap:anywhere}}
.user{{color:var(--user)}}.answer{{color:var(--answer)}}.reasoning{{color:var(--reasoning)}}.tool{{color:var(--tool)}}
.tool details{{margin-top:12px;border:1px solid #615044;border-radius:4px}}
.tool summary{{cursor:pointer;padding:10px 12px;overflow-wrap:anywhere}}
.tool summary:focus-visible{{outline:2px solid var(--tool);outline-offset:2px}}
.tool details[open] summary{{border-bottom:1px solid #615044}}
.tool details pre{{padding:12px}}
footer{{position:fixed;bottom:0;left:0;right:0;background:#262321;color:var(--program);border-top:1px solid #615044;padding:12px 24px;font-size:13px;max-height:30vh;overflow:auto}}
footer div{{max-width:1052px;margin:auto;white-space:pre-wrap;overflow-wrap:anywhere}}footer small{{display:block;color:var(--reasoning)}}
@media(max-width:600px){{header,main{{padding:16px}}main{{padding-bottom:10rem}}section{{padding-left:12px}}footer{{padding:10px 16px}}
.timing-help{{padding:0 16px}}#event-times{{margin-left:16px}}
.transcript-line,#event-times:checked ~ main .stream-event,#event-times:checked ~ main .status-event{{grid-template-columns:minmax(0,1fr);gap:2px}}
.line-time{{margin-top:6px}}.event-gap{{display:inline;margin-left:8px}}}}
@media print{{:root{{color-scheme:light;background:white;color:#222}}footer{{position:static}}main{{padding-bottom:16px}}}}
</style></head><body><header><h1>Conversation</h1>
<p>{_text(journal.stem)}</p></header>
<p class="timing-help">UTC timestamps mark the first event contributing to each logical line. Δ is the gap from the previous visible event (including status), not computation time. Lines from one event share its timing; wrapped rows are not new events. Individual events shows chunk/status timing. Journal wall clocks can change.</p>
<input type="checkbox" id="event-times"><label for="event-times"> Individual events</label>
<main aria-label="Terminal conversation">
"""
        encoded = header.encode("utf-8")
        self.offset = len(encoded)
        with self.path.open("wb") as handle:
            handle.write(encoded + self._footer())

    def _close_block(self) -> str:
        if self.open_kind is None:
            return ""
        self.open_kind = None
        return self._close_line() + "</pre></section>\n"

    def _close_line(self) -> str:
        if not self.line_open:
            return ""
        self.line_open = False
        return "</span></span>\n"

    def _timing(self, record: dict[str, Any]) -> str:
        value = record.get("timestamp")
        stamp = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                if math.isfinite(value):
                    stamp = datetime.fromtimestamp(value, timezone.utc)
            except (ValueError, OverflowError, OSError):
                pass
        if stamp is None:
            self.previous_timestamp = None
            self.updated = "Time unavailable"
            return 'Time unavailable<span class="event-gap">Δ unavailable</span>'
        timestamp = stamp.timestamp()
        self.updated = stamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
        gap = "Δ unavailable (first event or previous time unavailable)"
        if self.previous_timestamp is not None:
            delta = timestamp - self.previous_timestamp
            gap = (
                f"Δ {delta:.3f}s"
                if delta >= 0
                else "Δ unavailable (clock moved backwards)"
            )
        self.previous_timestamp = timestamp
        return (
            f'<time datetime="{stamp.isoformat(timespec="milliseconds")}">{self.updated}</time>'
            f'<span class="event-gap">{gap}</span>'
        )

    @staticmethod
    def _lines(text: Any, timing: str) -> str:
        return "\n".join(
            '<span class="transcript-line"><span class="line-time">'
            + timing
            + '</span><span class="line-content">'
            + _text(line)
            + "</span></span>"
            for line in str(text).split("\n")
        )

    def _stream(self, text: str, timing: str) -> str:
        if not text:
            return (
                '<span class="status-event"><span class="event-time">'
                + timing
                + '</span><span class="event-content">Empty stream event</span></span>'
            )
        output = []
        pieces = text.split("\n")
        for index, piece in enumerate(pieces):
            if index:
                output.append(self._close_line())
            if not piece and index == len(pieces) - 1:
                continue
            if not self.line_open:
                output.append(
                    '<span class="transcript-line"><span class="line-time">'
                    + timing
                    + '</span><span class="line-content">'
                )
                self.line_open = True
            output.append(
                '<span class="stream-event"><span class="event-time">'
                + timing
                + '</span><span class="event-content">'
                + _text(piece)
                + "</span></span>"
            )
        return "".join(output)

    def _footer(self) -> bytes:
        closing = (
            ("</span></span>\n" if self.line_open else "") + "</pre></section>\n"
            if self.open_kind is not None
            else ""
        )
        return (
            closing
            + '</main><footer aria-label="Current status"><div>'
            + _text(self.status)
            + (
                '<div class="line-time">' + self.status_timing + "</div>"
                if self.status_timing
                else ""
            )
            + "<small>"
            + _text(self.model)
            + (" · Updated " + _text(self.updated) if self.updated else "")
            + "</small></div></footer></body></html>\n"
        ).encode("utf-8")

    def append(self, record: dict[str, Any]) -> None:
        kind = record.get("type")
        if kind == "provider_tool_checkpoint":
            self.checkpoints[record["checkpoint_file"]] = record
            return
        if (
            kind in ("provider_tool", "tool_display")
            and record.get("display") is not False
        ):
            # The previous client displayed this exact durable path in its
            # completed-call notice. Upgrade that existing display position;
            # source checkpoint timing alone cannot establish terminal order.
            for line in str(record.get("tool", "")).splitlines():
                if line.startswith("Result checkpoint: "):
                    pointer = self.checkpoints.get(
                        line.removeprefix("Result checkpoint: ")
                    )
                    if pointer is not None:
                        record = {
                            **pointer,
                            "type": "tool_result_display",
                            "timestamp": record.get("timestamp"),
                        }
                        kind = "tool_result_display"
                    break
        fragment = ""
        if kind == "conversation":
            self.model = str(record.get("model") or "")
        elif kind == "display_options":
            self.show_reasoning = bool(record.get("show_reasoning", True))
            return
        elif kind == "user_message":
            timing = self._timing(record)
            fragment = (
                self._close_block()
                + '<section class="user"><h2>You</h2><pre class="timed-lines">'
            )
            fragment += (
                self._lines(record.get("prompt", ""), timing) + "</pre></section>\n"
            )
            self.status_timing = timing
            self.status = "Waiting for response"
        elif kind == "assistant_stream":
            event = record.get("event_type")
            if event not in ("text", "reasoning") or (
                event == "reasoning" and not self.show_reasoning
            ):
                return
            role = "answer" if event == "text" else "reasoning"
            turn_id = record.get("turn_id")
            if self.open_kind != role or self.turn_id != turn_id:
                fragment = self._close_block()
                label = "LLM" if role == "answer" else "LLM · Reasoning"
                fragment += f'<section class="{role}"><h2>{label}</h2><pre class="timed-lines stream-lines">'
                self.open_kind = role
                self.turn_id = turn_id
            timing = self._timing(record)
            fragment += self._stream(str(record.get("chunk", "")), timing)
            self.status_timing = timing
            self.status = (
                "Receiving reply" if role == "answer" else "Receiving reasoning"
            )
        elif kind == "program_message":
            timing = self._timing(record)
            fragment = (
                self._close_block()
                + '<div class="program">'
                + self._lines(record.get("text", ""), timing)
                + "</div>\n"
            )
        elif kind == "tool_result_display":
            # Import at use time: the checkpoint reader also uses the journal.
            # Read the verified sidecar only once per returned call, never for
            # each streamed token. Keep the full output out of the JSONL mirror.
            from .openwebui_recovery import _load_checkpoint

            checkpoint = _load_checkpoint(self.conversation_id, record)
            identity = (checkpoint["provider_run_id"], checkpoint["call_id"])
            if identity in self.tool_results:
                return
            timing = self._timing(record)
            fragment = (
                self._close_block()
                + '<section class="tool"><h2>'
                + _text(checkpoint["name"])
                + ' · returned result</h2><div class="line-time">'
                + timing
                + '</div><details class="tool-metadata">'
                + '<summary>Call metadata</summary><pre class="timed-lines">'
                + self._lines(
                    checkpoint_notice(checkpoint, record, preview=False), timing
                )
                + '</pre></details><details class="tool-result"><summary>Full result · '
                + _text(checkpoint["name"])
                + '</summary><pre class="timed-lines">'
                + self._lines(tool_output_text(checkpoint["output"]), timing)
                + "</pre></details></section>\n"
            )
            self.tool_results.add(identity)
        elif kind in ("provider_tool", "tool_display"):
            if record.get("display") is False:
                return
            timing = self._timing(record)
            fragment = (
                self._close_block()
                + '<section class="tool"><h2>Tool</h2><pre class="timed-lines">'
                + self._lines(record.get("tool", ""), timing)
                + "</pre></section>\n"
            )
        elif kind in ("provider_status", "status_display"):
            if record.get("display") is False:
                return
            timing = self._timing(record)
            self.status = "Open WebUI │ " + str(record.get("status") or "")
            self.status_timing = timing
            fragment = (
                '<span class="status-event"><span class="event-time">'
                + timing
                + '</span><span class="event-content">'
                + _text(self.status)
                + "</span></span>"
            )
        elif kind in ("turn_cancelled", "turn_error"):
            message = (
                cancellation_notice(record)
                if kind == "turn_cancelled"
                else "Model error: " + str(record.get("error") or "")
            )
            timing = self._timing(record)
            fragment = (
                self._close_block()
                + '<div class="program">'
                + self._lines(message, timing)
                + "</div>\n"
            )
            self.status = message
            self.status_timing = timing
        elif kind in ("turn_completed", "recovery_completed"):
            timing = self._timing(record)
            fragment = (
                self._close_block()
                + '<div class="program">'
                + self._lines("Response completed", timing)
                + "</div>\n"
            )
            self.status = "Response completed"
            self.status_timing = timing
        elif kind == "provider_error":
            timing = self._timing(record)
            self.status = str(record.get("error") or "Provider error")
            self.status_timing = timing
            fragment = (
                self._close_block()
                + '<div class="program">'
                + self._lines(self.status, timing)
                + "</div>\n"
            )
        else:
            # Request metadata and checkpoint payloads are not terminal output.
            return
        encoded = fragment.encode("utf-8")
        with self.path.open("r+b") as handle:
            # Replace only the small closing/footer tail. An open stream block
            # is closed in every on-disk snapshot, then continued on the next event.
            handle.seek(self.offset)
            handle.write(encoded)
            self.offset += len(encoded)
            handle.write(self._footer())
            handle.truncate()
            handle.flush()


def update_html_report(journal: Path, record: dict[str, Any] | None = None) -> None:
    """Called under the journal lock, after its record has been flushed."""
    if journal in _disabled:
        return
    try:
        report = _reports.get(journal)
        if report is None:
            # A rebuild must not replace a readable prior report until all
            # durable journal records have been rendered successfully.
            with tempfile.NamedTemporaryFile(
                dir=journal.parent,
                prefix=f".{journal.stem}-",
                suffix=".html.tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                report = _HTMLReport(journal, path=temporary_path)
                with journal.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            saved = json.loads(line)
                        except json.JSONDecodeError:
                            if not line.endswith("\n") and handle.read(1) == "":
                                break
                            raise
                        if isinstance(saved, dict):
                            report.append(saved)
                os.replace(temporary_path, report_path(journal))
                report.path = report_path(journal)
            finally:
                temporary_path.unlink(missing_ok=True)
            _reports[journal] = report
            # The newly flushed record is already included in that rebuild.
        elif record is not None:
            report.append(record)
    except (OSError, ValueError) as exc:
        _reports.pop(journal, None)
        _disabled.add(journal)
        echo_program(
            f"HTML report unavailable: {report_path(journal)} ({type(exc).__name__}). "
            "The conversation journal remains active.",
            err=True,
        )


def ensure_html_report(journal: Path) -> None:
    # An explicit reopen may retry a previously unavailable report. Stream
    # events themselves never trigger automatic retries after an I/O failure.
    _disabled.discard(journal)
    update_html_report(journal)
