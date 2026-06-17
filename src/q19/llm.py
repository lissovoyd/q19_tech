"""Phase 3 LLM client — two backend options (Anthropic SDK or local Ollama),
prompt-hash disk cache, retry + backoff, token tracking, and real cost
reporting for Anthropic or hypothetical cost reporting for Ollama.

Cache: outputs/llm_cache/<sha256>.json — a hit never touches the network,
so re-runs are free and deterministic. Every call's payload + response is
stored, so the cache is also an audit trail.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "outputs" / "llm_cache"

load_dotenv(ROOT / ".env")

# ===========================================================================
# BACKEND CONFIGURATION — one ACTIVE, one DISABLED.
# Switch by changing PROVIDER and the matching model/key lines.
# ===========================================================================

# OPTION A — DISABLED: Anthropic API (claude-haiku-4-5).
# Real cost reported (not hypothetical). Requires ANTHROPIC_API_KEY env var.
# PROVIDER = "anthropic"
# MODEL = "claude-haiku-4-5-20251001"
# _COST_PER_1M_INPUT = 0.80   # USD — Haiku actual pricing
# _COST_PER_1M_OUTPUT = 4.00  # USD — Haiku actual pricing
# _COST_IS_REAL = True

# OPTION B — ACTIVE: local Ollama (Qwen2.5 7B quantised).
# No network dependency, fully offline. Cost shown is hypothetical (GPT-4o-mini rates).
PROVIDER = "ollama"
MODEL = "qwen2.5:7b-instruct-q4_K_M"
_COST_PER_1M_INPUT = 0.15   # USD — GPT-4o-mini hypothetical reference
_COST_PER_1M_OUTPUT = 0.60  # USD — GPT-4o-mini hypothetical reference
_COST_IS_REAL = False

# ===========================================================================

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds; doubles each retry

_usage: dict[str, int] = {
    "calls": 0,
    "cache_hits": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    messages: list[dict[str, str]],
    *,
    model: str = MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Send a chat request; return {"text": str, "usage": dict, "cached": bool}.

    Checks the disk cache first. On a miss, calls the configured backend with
    retry + exponential backoff and writes the response to the cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_payload = {
        "provider": PROVIDER,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    key = _cache_key(cache_payload)
    cache_path = CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        _accumulate(stored["usage"], cache_hit=True)
        return {"text": stored["text"], "usage": stored["usage"], "cached": True}

    if PROVIDER == "anthropic":
        response = _call_anthropic(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        response = _call_ollama(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    cache_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    _accumulate(response["usage"], cache_hit=False)
    return {"text": response["text"], "usage": response["usage"], "cached": False}


def usage_summary() -> dict[str, Any]:
    """Return accumulated token usage + cost (real for Anthropic, hypothetical for Ollama)."""
    inp = _usage["prompt_tokens"]
    out = _usage["completion_tokens"]
    cost = (inp / 1_000_000) * _COST_PER_1M_INPUT + (out / 1_000_000) * _COST_PER_1M_OUTPUT
    return {
        **_usage,
        "total_tokens": inp + out,
        "cost_usd": round(cost, 6),
        "cost_is_real": _COST_IS_REAL,
        "model": MODEL,
        "provider": PROVIDER,
    }


def reset_usage() -> None:
    _usage.update(calls=0, cache_hits=0, prompt_tokens=0, completion_tokens=0)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _call_anthropic(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    import anthropic  # local import — only needed when this backend is active
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    system_msgs = [m["content"] for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] != "system"]
    system_param = system_msgs[0] if system_msgs else anthropic.NOT_GIVEN

    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                messages=user_msgs,
                system=system_param,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.content[0].text.strip()
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
            return {"text": text, "usage": usage}
        except Exception as exc:
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Anthropic call failed after {_RETRIES} attempts: {last_exc}") from last_exc


def _call_ollama(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(
                    f"{_OLLAMA_BASE_URL}/chat/completions",
                    headers={"Authorization": "Bearer ollama",
                             "Content-Type": "application/json"},
                    content=json.dumps(payload).encode(),
                )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            usage = {
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
            return {"text": text, "usage": usage}
        except Exception as exc:
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Ollama call failed after {_RETRIES} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _accumulate(usage: dict, *, cache_hit: bool) -> None:
    _usage["calls"] += 1
    _usage["cache_hits"] += int(cache_hit)
    _usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
    _usage["completion_tokens"] += usage.get("completion_tokens", 0)
