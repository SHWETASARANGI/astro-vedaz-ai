"""
Shared I/O helpers for the chat pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

Chat = dict[str, Any]


def load_jsonl(path: str | Path) -> list[Chat]:
    """
    Load a .jsonl file into a list of dicts.

    Skips blank lines. Raises a clear error (with line number) on malformed
    JSON instead of letting json.JSONDecodeError bubble up unexplained.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    chats: list[Chat] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chats.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed JSON in {path} at line {line_no}: {e}"
                ) from e

    logger.info("Loaded %d records from %s", len(chats), path)
    return chats


def save_jsonl(chats: Iterable[Chat], path: str | Path) -> None:
    """
    Write an iterable of dicts to a .jsonl file, creating parent
    directories if they don't exist yet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chat in chats:
            f.write(json.dumps(chat, ensure_ascii=False))
            f.write("\n")
            count += 1

    logger.info("Wrote %d records to %s", count, path)


def get_chat_text(chat: Chat) -> str:
    """
    Concatenate all message contents in a chat into a single string.
    Centralized here so checker.py/evaluator.py don't each reimplement it
    slightly differently (and so a missing "content" key fails loudly,
    not silently with a KeyError deep in a loop).
    """
    messages = chat.get("messages", [])
    parts = []
    for i, msg in enumerate(messages):
        if "content" not in msg:
            raise KeyError(f"Message {i} is missing a 'content' field: {msg}")
        parts.append(str(msg["content"]))
    return " ".join(parts)