"""
llm.py — Thin wrapper around the Gemini API via its OpenAI-compatible endpoint.

No other files need to change — this is the only file that knows about the
underlying LLM provider. To switch back to Together AI (or any other
OpenAI-compatible provider), update the three constants below and the .env key.

Free Gemini API key: https://aistudio.google.com/app/apikey
Free tier limits (gemini-2.0-flash): 15 RPM · 1 500 RPD · 1M TPM
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

# Provider config —
DEFAULT_MODEL   = "gemini-2.0-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_ENV_KEY_NAME   = "GEMINI_API_KEY"

DEFAULT_TIMEOUT        = 60   # seconds per request
MAX_RETRIES            = 3
RETRY_BACKOFF_SECONDS  = 5    # Gemini free tier needs a longer back-off than Together AI


def _build_client() -> OpenAI:
    api_key = os.getenv(_ENV_KEY_NAME)
    if not api_key:
        raise EnvironmentError(
            f"{_ENV_KEY_NAME} is not set. Add it to your .env file "
            "(see .env.example) or export it in your shell.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT)


# Lazy singleton — importing this module never makes a network call.
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def generate_response(
    prompt: str,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """
    Generate a single completion for `prompt` using the Gemini API.

    Retries on rate-limit and timeout errors with exponential-ish back-off.
    Raises on hard API errors so callers can decide how to handle failure.

    Args:
        prompt:        The user-turn content.
        system_prompt: Optional system instruction (maps to Gemini's system_instruction).
        model:         Gemini model string. Defaults to gemini-2.0-flash (free tier).
        temperature:   Sampling temperature (0.0–1.0).
        max_tokens:    Cap on output tokens. None = model default.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    client = get_client()

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Gemini returned an empty response")
            return content

        except RateLimitError as e:
            # Free tier: 15 requests/minute — back off and retry
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Rate limit hit (attempt %d/%d) — waiting %ds before retry. "
                "Free tier allows 15 RPM; consider adding delays between calls.",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)

        except APITimeoutError as e:
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Timeout on attempt %d/%d — retrying in %ds",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)

        except APIError as e:
            # Non-retryable: bad request, invalid key, model not found, etc.
            logger.error("Non-retryable Gemini API error: %s", e)
            raise

    logger.error("All %d attempts failed for prompt: %.80s", MAX_RETRIES, prompt)
    raise last_error