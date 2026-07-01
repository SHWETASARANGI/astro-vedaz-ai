"""
checker.py — Validates, deduplicates, and safety-checks generated chat data,
then splits it into train/test sets.

Usage:
    python -m src.checker
    python -m src.checker --input data/input_chats.jsonl --test-size 0.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from safety import check_safety
from utils import Chat, get_chat_text, load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_ROLE_SEQUENCE_START = "system"
TURN_ROLES = ("user", "assistant")


def validate_chat(chat: Chat) -> bool:
    """
    A chat is valid if it:
    - has at least one message
    - starts with a system message
    - alternates user/assistant strictly after that
    """
    messages = chat.get("messages", [])
    if not messages:
        return False
    if messages[0].get("role") != VALID_ROLE_SEQUENCE_START:
        return False

    expected = "user"
    for msg in messages[1:]:
        if msg.get("role") != expected:
            return False
        expected = "assistant" if expected == "user" else "user"
    return True


def count_words(chat: Chat) -> int:
    return len(get_chat_text(chat).split())


def hash_chat(chat: Chat) -> str:
    return hashlib.md5(get_chat_text(chat).encode("utf-8")).hexdigest()


def find_duplicates(chats: list[Chat]) -> list[int]:
    """Returns indices of chats that are exact-text duplicates of an earlier chat."""
    seen: dict[str, int] = {}
    duplicates: list[int] = []
    for idx, chat in enumerate(chats):
        key = hash_chat(chat)
        if key in seen:
            duplicates.append(idx)
        else:
            seen[key] = idx
    return duplicates


def run_checker(
    input_file: str | Path,
    output_dir: str | Path = "outputs",
    data_dir: str | Path = "data",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_invalid: bool = True,
    drop_unsafe: bool = True,
    drop_duplicates: bool = True,
) -> dict[str, Any]:
    """
    Run validation, safety checks, and dedup on `input_file`, then write a
    train/test split (of the *clean* chats, unless the drop_* flags are
    turned off) plus a JSON report summarizing what was found.
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    chats = load_jsonl(input_file)
    if not chats:
        raise ValueError(f"No chats loaded from {input_file}")

    invalid_idx: list[int] = []
    unsafe_records: list[dict[str, Any]] = []
    lengths: list[int] = []

    for i, chat in enumerate(chats):
        if not validate_chat(chat):
            invalid_idx.append(i)

        lengths.append(count_words(chat))

        result = check_safety(get_chat_text(chat))
        if not result.safe:
            categories = sorted({v.category for v in result.violations if not v.negated})
            unsafe_records.append({"index": i, "violations": categories})

    unsafe_idx = {r["index"] for r in unsafe_records}
    duplicate_idx = set(find_duplicates(chats))

    # Decide which indices to drop from the train/test split
    drop_idx: set[int] = set()
    if drop_invalid:
        drop_idx |= set(invalid_idx)
    if drop_unsafe:
        drop_idx |= unsafe_idx
    if drop_duplicates:
        drop_idx |= duplicate_idx

    clean_chats = [c for i, c in enumerate(chats) if i not in drop_idx]

    if len(clean_chats) < 2:
        logger.warning(
            "Only %d clean chats remain after filtering — skipping train/test split.",
            len(clean_chats),
        )
        train, test = clean_chats, []
    else:
        train, test = train_test_split(
            clean_chats, test_size=test_size, random_state=random_state
        )

    save_jsonl(train, data_dir / "train.jsonl")
    save_jsonl(test, data_dir / "test.jsonl")

    report = {
        "total_chats": len(chats),
        "invalid_chats": len(invalid_idx),
        "unsafe_chats": len(unsafe_records),
        "duplicate_chats": len(duplicate_idx),
        "clean_chats_used_for_split": len(clean_chats),
        "train_size": len(train),
        "test_size": len(test),
        "avg_words": (sum(lengths) / len(lengths)) if lengths else 0,
        "unsafe_details": unsafe_records,
        "invalid_indices": invalid_idx,
        "duplicate_indices": sorted(duplicate_idx),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "checker_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Report written to %s", report_path)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate, dedupe, and safety-check chat data.")
    parser.add_argument("--input", default="data/input_chats.jsonl", help="Path to input .jsonl file")
    parser.add_argument("--output-dir", default="outputs", help="Directory for checker_report.json")
    parser.add_argument("--data-dir", default="data", help="Directory for train/test split output")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of clean data used for test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the train/test split")
    parser.add_argument("--keep-invalid", action="store_true", help="Include invalid chats in train/test split")
    parser.add_argument("--keep-unsafe", action="store_true", help="Include unsafe chats in train/test split")
    parser.add_argument("--keep-duplicates", action="store_true", help="Include duplicate chats in train/test split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_checker(
        input_file=args.input,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_invalid=not args.keep_invalid,
        drop_unsafe=not args.keep_unsafe,
        drop_duplicates=not args.keep_duplicates,
    )