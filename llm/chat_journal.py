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
) -> Path:
    """Append one complete JSON object to a conversation journal."""
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
    with _lock:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
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
