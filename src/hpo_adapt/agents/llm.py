"""Minimal LLM client abstraction for the Setup Agents (spec.md 模块一 / 模块四).

API keys are read ONLY from environment variables (never hardcoded):
  - OPENAI_API_KEY   (required to enable the LLM backend)
  - OPENAI_BASE_URL  (optional, defaults to https://api.openai.com/v1)
  - OPENAI_MODEL     (optional, defaults to gpt-4o-mini)

When no key is configured, the deterministic heuristic agents are used, so the
full closed loop stays runnable and testable offline.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def is_llm_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


class OpenAICompatClient:
    """OpenAI-compatible chat-completions client using only the standard library."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {body}") from exc


def extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM reply (tolerates code fences)."""
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                cleaned = candidate
                break
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("no JSON object found in LLM output")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON in LLM output: {exc}") from exc
