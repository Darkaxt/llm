from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import user_dir

_lock = threading.Lock()


def journal_dir() -> Path:
    configured = os.environ.get("LLM_CHAT_EXPORT_DIR")
    path = Path(configured).expanduser() if configured else user_dir() / "conversations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def journal_path(conversation_id: str) -> Path:
    return journal_dir() / f"{conversation_id}.jsonl"


def append_record(
    conversation_id: str,
    record: dict[str, Any],
    *,
    timestamp: float | None = None,
    durable: bool = False,
) -> Path:
    """Append one JSON object, optionally syncing a checkpoint pointer to disk."""
    path = journal_path(conversation_id)
    payload = {
        "timestamp": timestamp if timestamp is not None else time.time(),
        "conversation_id": conversation_id,
        **record,
    }
    line = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    encoded = (line + "\n").encode("utf-8")
    with _lock:
        # A process may have died midway through the final JSONL record.
        # Remove ONLY that incomplete tail before appending the next record,
        # so a recovery attempt does not turn it into interior corruption.
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            if position:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    while position:
                        start = max(0, position - 4096)
                        handle.seek(start)
                        chunk = handle.read(position - start)
                        last_newline = chunk.rfind(b"\n")
                        if last_newline >= 0:
                            handle.truncate(start + last_newline + 1)
                            break
                        position = start
                    else:
                        handle.truncate(0)
            handle.seek(0, os.SEEK_END)
            handle.write(encoded)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
    return path


def ensure_session(
    conversation_id: str,
    *,
    model: str,
    name: str | None = None,
) -> Path:
    path = journal_path(conversation_id)
    with _lock:
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            return path

        payload = {
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "type": "conversation",
            "model": model,
            "name": name,
            "format": "llm-chat-journal-v2",
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            handle.write("\n")
            handle.flush()
    return path
