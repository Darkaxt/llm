"""Opt-in APISIX HTTP body capture and offline, payload-free timing summaries."""

import base64
import datetime
import json
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import closing
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx2


class _Capture:
    def __init__(
        self, directory: str | Path, request: httpx2.Request, profile: str | None
    ):
        self.started = time.monotonic()
        self.finished = False
        trace_id = uuid.uuid4().hex
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"apisix-{trace_id}.jsonl"
        with self.path.open("x", encoding="utf-8") as handle:
            handle.write("")
        self.write(
            "request",
            schema_version=1,
            trace_id=trace_id,
            profile=profile,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            method=request.method,
            content_type=request.headers.get("content-type"),
            body_b64=base64.b64encode(request.content).decode("ascii"),
        )

    def write(self, kind: str, **values: Any) -> None:
        record = {
            "type": kind,
            "elapsed_seconds": time.monotonic() - self.started,
            **values,
        }
        # Close after every append: a cancelled/abandoned parser cannot retain the
        # artifact handle. Opening happens after the arrival timestamp is taken.
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self, outcome: str, error: BaseException | None = None) -> None:
        if not self.finished:
            self.finished = True
            self.write(
                "end",
                outcome=outcome,
                error_type=type(error).__name__ if error else None,
            )


class _CaptureStream(httpx2.SyncByteStream):
    def __init__(self, stream: httpx2.SyncByteStream, capture: _Capture):
        self.stream = stream
        self.capture = capture

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self.stream:
                self.capture.write(
                    "chunk", body_b64=base64.b64encode(chunk).decode("ascii")
                )
                yield chunk
        except GeneratorExit:
            self.capture.finish("closed")
            raise
        except BaseException as ex:
            self.capture.finish(
                "cancelled" if isinstance(ex, KeyboardInterrupt) else "error", ex
            )
            raise
        else:
            self.capture.finish("eof")

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            self.capture.finish("closed")


class CaptureClient(httpx2.Client):
    """Capture response bytes before HTTP content decoding and SDK SSE parsing."""

    def __init__(
        self, directory: str | Path, *, profile: str | None = None, **kwargs: Any
    ):
        self.capture_directory = directory
        self.capture_profile = profile
        super().__init__(event_hooks={"response": [self._capture_response]}, **kwargs)

    def send(self, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
        capture = _Capture(self.capture_directory, request, self.capture_profile)
        request.extensions["llm_apisix_capture"] = capture
        try:
            return super().send(request, **kwargs)
        except BaseException as ex:
            capture.finish(
                "cancelled" if isinstance(ex, KeyboardInterrupt) else "error", ex
            )
            raise

    def _capture_response(self, response: httpx2.Response) -> None:
        capture = response.request.extensions["llm_apisix_capture"]
        capture.write(
            "response",
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            content_encoding=response.headers.get("content-encoding", ""),
            upstream_request_id=response.headers.get("x-request-id"),
        )
        assert isinstance(response.stream, httpx2.SyncByteStream)
        response.stream = _CaptureStream(response.stream, capture)


class _Frames:
    def __init__(self) -> None:
        self.pending = b""
        self.lines: list[bytes] = []
        self.first_line = True

    def feed(self, chunk: bytes, *, final: bool = False) -> Iterator[list[bytes]]:
        self.pending += chunk
        while match := re.search(b"[\r\n]", self.pending):
            pos = match.start()
            if self.pending[pos:] == b"\r" and not final:
                break  # A CRLF may be split across HTTP chunks.
            width = 2 if self.pending[pos : pos + 2] == b"\r\n" else 1
            line, self.pending = self.pending[:pos], self.pending[pos + width :]
            if self.first_line:
                line = line.removeprefix(b"\xef\xbb\xbf")
                self.first_line = False
            if line:
                self.lines.append(line)
            elif self.lines:
                yield self.lines
                self.lines = []


def summarize_trace(path: str | Path) -> dict[str, Any]:
    """Replay HTTP decompression offline; timestamps refer to captured arrivals."""
    rows: list[dict[str, Any]] = []
    partial_record = False
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    partial_record = True
                else:
                    raise ValueError("Invalid trace record") from None
    if not rows or rows[0].get("type") != "request":
        raise ValueError("Not an APISIX capture")
    start = rows[0]
    response = next((row for row in rows if row["type"] == "response"), {})
    end: dict[str, Any] = next(
        (row for row in reversed(rows) if row["type"] == "end"), {}
    )
    chunks = [row for row in rows if row["type"] == "chunk"]
    elapsed = end.get("elapsed_seconds")
    arrivals = [row["elapsed_seconds"] for row in chunks]
    boundaries = [0, *arrivals, *([elapsed] if elapsed is not None else [])]
    report: dict[str, Any] = {
        "schema_version": 1,
        "trace_id": start.get("trace_id"),
        "profile": start.get("profile"),
        "started_utc": start.get("started_utc"),
        "status_code": response.get("status_code"),
        "upstream_request_id": response.get("upstream_request_id"),
        "outcome": end.get("outcome", "unfinished"),
        "error_type": end.get("error_type"),
        "duration_seconds": elapsed,
        "observed_until_seconds": rows[-1]["elapsed_seconds"],
        "first_byte_seconds": arrivals[0] if arrivals else None,
        "max_no_byte_gap_seconds": max(
            (b - a for a, b in pairwise(boundaries)), default=None
        ),
        "raw_body_bytes": sum(len(base64.b64decode(row["body_b64"])) for row in chunks),
        "body_chunks": len(chunks),
        "first_event_seconds": None,
        "first_reasoning_seconds": None,
        "first_answer_seconds": None,
        "reasoning_events": 0,
        "answer_events": 0,
        "tool_events": 0,
        "reasoning_event_span_seconds": None,
        "max_inter_reasoning_event_gap_seconds": None,
        "finish_reasons": [],
        "usage": None,
        "done_received": False,
        "eof_observed": end.get("outcome") == "eof",
        "incomplete_capture_record": partial_record,
        "incomplete_sse_tail": False,
        "non_json_events": 0,
        "provider_error_events": 0,
        "malformed_payload_events": 0,
        "decoding_error_type": None,
        "timing_note": "Client-observed delivery times, including consumer backpressure; reasoning event span is not internal computation time.",
    }
    reasoning_times: list[float] = []

    def inspect_payload(payload: Any, at: float) -> None:
        if not isinstance(payload, dict):
            return
        reasoning = answer = tools = False
        choices = payload.get("choices") or []
        if not isinstance(choices, list):
            report["malformed_payload_events"] += 1
            choices = []
        for choice in choices:
            if not isinstance(choice, dict):
                report["malformed_payload_events"] += 1
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                report["malformed_payload_events"] += 1
                continue
            reasoning |= bool(delta.get("reasoning_content") or delta.get("reasoning"))
            answer |= bool(delta.get("content"))
            tools |= bool(delta.get("tool_calls"))
            finish = choice.get("finish_reason")
            if finish and finish not in report["finish_reasons"]:
                report["finish_reasons"].append(finish)
        delta = payload.get("delta") or {}
        if not isinstance(delta, dict):
            report["malformed_payload_events"] += 1
            delta = {}
        reasoning |= delta.get("type") == "thinking_delta" and bool(
            delta.get("thinking")
        )
        answer |= delta.get("type") == "text_delta" and bool(delta.get("text"))
        if reasoning:
            reasoning_times.append(at)
            report["reasoning_events"] += 1
            if report["first_reasoning_seconds"] is None:
                report["first_reasoning_seconds"] = at
        if answer:
            report["answer_events"] += 1
            if report["first_answer_seconds"] is None:
                report["first_answer_seconds"] = at
        report["tool_events"] += int(tools)
        report["provider_error_events"] += int(bool(payload.get("error")))
        usage = payload.get("usage")
        if isinstance(usage, dict):
            # Summary exports contain known numeric usage fields, never arbitrary
            # provider text. The full usage object remains in the local raw trace.
            counts = {
                key: usage[key]
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "input_tokens",
                    "output_tokens",
                )
                if isinstance(usage.get(key), (int, float))
            }
            for key in ("prompt_tokens_details", "completion_tokens_details"):
                details = usage.get(key)
                if isinstance(details, dict):
                    counts[key] = {
                        name: value
                        for name, value in details.items()
                        if isinstance(value, (int, float))
                    }
            report["usage"] = counts

    class Replay(httpx2.SyncByteStream):
        at: float = 0

        def __iter__(self) -> Iterator[bytes]:
            for row in chunks:
                self.at = row["elapsed_seconds"]
                yield base64.b64decode(row["body_b64"])

    replay = Replay()
    content_encoding = response.get("content_encoding", "")
    # Public HTTPX response decoding supports the same compressed encodings as
    # the live SDK. Neither bytes nor model parameters are changed during capture.
    with closing(
        httpx2.Response(
            200, headers={"content-encoding": content_encoding}, stream=replay
        )
    ) as decoded:

        def decoded_chunks() -> Iterator[bytes]:
            try:
                yield from decoded.iter_bytes()
            except httpx2.DecodingError as ex:
                report["decoding_error_type"] = type(ex).__name__

        if "text/event-stream" in response.get("content_type", "").lower():
            frames = _Frames()

            def inspect_frame(lines: list[bytes]) -> None:
                if report["first_event_seconds"] is None:
                    report["first_event_seconds"] = replay.at
                data = b"\n".join(
                    line[5:].removeprefix(b" ")
                    for line in lines
                    if line.startswith(b"data:")
                )
                if data.strip() == b"[DONE]":
                    report["done_received"] = True
                elif data:
                    try:
                        payload = json.loads(data)
                    except (ValueError, UnicodeDecodeError):
                        report["non_json_events"] += 1
                    else:
                        inspect_payload(payload, replay.at)

            for chunk in decoded_chunks():
                for frame in frames.feed(chunk):
                    inspect_frame(frame)
            for frame in frames.feed(b"", final=True):
                inspect_frame(frame)
            report["incomplete_sse_tail"] = bool(frames.pending or frames.lines)
        else:
            body = b"".join(decoded_chunks())
            try:
                inspect_payload(json.loads(body), replay.at)
            except (ValueError, UnicodeDecodeError):
                pass
    if reasoning_times:
        report["reasoning_event_span_seconds"] = (
            reasoning_times[-1] - reasoning_times[0]
        )
        report["max_inter_reasoning_event_gap_seconds"] = max(
            (b - a for a, b in pairwise(reasoning_times)), default=None
        )
    return report
