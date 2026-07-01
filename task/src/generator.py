"""
generator.py — Generates synthetic astrology assistant chats via the LLM,
validates structure, and filters out unsafe content before saving.

Usage:
    python -m src.generator
    python -m src.generator --per-topic 3 --output data/generated_chats.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re

from checker import validate_chat
from llm import generate_response
from safety import check_safety
from utils import get_chat_text, save_jsonl
from language import HINGLISH_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


TOPICS = [
    "career growth slow chal rahi hai",
    "government exam stress",
    "job loss anxiety",
    "late marriage concern",
    "love marriage vs arranged marriage",
    "relationship confusion",
    "family pressure for shaadi",
    "moving abroad uncertainty",
    "business slowdown",
    "financial worries",
    "career switch confusion",
    "startup failure fear",
    "higher studies abroad",
    "long distance relationship",
    "parent-child conflict",
]

SYSTEM_PROMPT = HINGLISH_SYSTEM_PROMPT

PROMPT_TEMPLATE = """
Generate a realistic astrology assistant chat conversation.

Audience:
Indian astrology-app users.

Language:
Natural Hinglish.

Examples of style:

User:
Meri job growth kaafi slow chal rahi hai.

Assistant:
Main samajh sakta hoon ki career delays frustrating lag sakte hain. Astrology kuch patterns aur timings par reflection dene mein madad kar sakti hai, lekin future ko certainty ke saath predict nahi karti.

Requirements:

The chat must contain:

- one system message
- alternating user and assistant messages
- at least 2 user turns
- at least 2 assistant turns

Assistant behavior:

✓ warm
✓ empathetic
✓ respectful
✓ conversational
✓ realistic

Assistant must NEVER:

✗ predict death
✗ predict illness
✗ diagnose medical conditions
✗ guarantee money
✗ guarantee marriage
✗ guarantee career success
✗ pressure users into buying remedies
✗ create fear
✗ claim astrology is certain

The assistant should regularly remind users that:

- astrology offers reflection, not certainty
- personal actions matter
- important medical concerns require doctors
- financial outcomes cannot be guaranteed

Topic:
{topic}

Return ONLY valid JSON.

Format:

{{
  "messages": [
    {{
      "role": "system",
      "content": "You are a warm astrology guide."
    }},
    {{
      "role": "user",
      "content": "..."
    }},
    {{
      "role": "assistant",
      "content": "..."
    }}
  ]
}}

No markdown.
No explanations.
Only JSON.
"""

INDIAN_CONTEXT_HINTS = [
    "middle-class working professional",
    "engineering student",
    "government exam aspirant",
    "recent graduate",
    "small business owner",
    "IT employee",
    "working woman facing family pressure for marriage",
]

# Some models wrap JSON in ```json ... ``` even when told not to — strip that defensively.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(raw: str) -> dict | None:
    """Best-effort parse of a model response into a dict, tolerating code fences/whitespace."""
    cleaned = _CODE_FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block in case there's stray text around it.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


def generate_chat(topic: str) -> dict | None:
    """Generate one chat for `topic`. Returns None on any failure (logged, not raised)."""
    prompt = PROMPT_TEMPLATE.format(
    topic=f"{topic} | User profile: {random.choice(INDIAN_CONTEXT_HINTS)}"
)
    try:
        response = generate_response(prompt, system_prompt=SYSTEM_PROMPT)
    except Exception as e:
        logger.error("LLM call failed for topic %r: %s", topic, e)
        return None

    chat = _extract_json(response)
    if chat is None:
        logger.warning("Could not parse JSON for topic %r. Raw response: %.200s", topic, response)
        return None
    return chat


def generate_dataset(topics: list[str], per_topic: int = 1) -> tuple[list[dict], dict]:
    """
    Generate `per_topic` chats for each topic, filtering out structurally
    invalid or unsafe ones. Returns (accepted_chats, stats).
    """
    accepted: list[dict] = []
    stats = {
        "attempted": 0,
        "parse_failed": 0,
        "invalid_structure": 0,
        "unsafe": 0,
        "accepted": 0,
        "rejected_examples": [],
    }

    for topic in topics:
        for _ in range(per_topic):
            stats["attempted"] += 1
            chat = generate_chat(topic)

            if chat is None:
                stats["parse_failed"] += 1
                continue

            if not validate_chat(chat):
                stats["invalid_structure"] += 1
                logger.warning("Invalid structure for topic %r", topic)
                continue

            text = get_chat_text(chat)
            safety = check_safety(text)
            if not safety.safe:
                stats["unsafe"] += 1
                categories = sorted({v.category for v in safety.violations if not v.negated})
                logger.warning("Unsafe chat dropped for topic %r: %s", topic, categories)
                stats["rejected_examples"].append({"topic": topic, "violations": categories})
                continue

            accepted.append(chat)
            stats["accepted"] += 1

    return accepted, stats


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic astrology chats.")
    parser.add_argument("--output", default="data/generated_chats.jsonl")
    parser.add_argument("--per-topic", type=int, default=1, help="Number of chats to generate per topic")
    parser.add_argument("--topics", nargs="*", default=None, help="Override the default topic list")
    args = parser.parse_args()

    topics = args.topics if args.topics else TOPICS
    accepted, stats = generate_dataset(topics, per_topic=args.per_topic)

    save_jsonl(accepted, args.output)

    logger.info("Generation stats: %s", json.dumps(stats, indent=2))
    print(f"Saved {len(accepted)}/{stats['attempted']} chats to {args.output}")


if __name__ == "__main__":
    main()