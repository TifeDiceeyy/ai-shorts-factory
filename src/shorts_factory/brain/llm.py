"""Optional LLM adapter (OpenAI-compatible chat completions).

Works with OpenAI, and with any OpenAI-compatible endpoint (Ollama, LM
Studio, vLLM, etc.). Configure with env vars (see config.py):

    BRAIN_LLM_BASE_URL   default https://api.openai.com/v1
    BRAIN_LLM_API_KEY    default empty
    BRAIN_LLM_MODEL      default gpt-4o-mini

The brain itself never needs this — retrieval is fully local. This is only
used to polish scripts when you want a model to rewrite book facts.
"""
from __future__ import annotations

import json
import urllib.request

from . import config


def llm_available() -> bool:
    return bool(config.LLM_API_KEY.strip())


def llm_chat(prompt: str, json_mode: bool = False) -> object:
    """Send one chat-completions request and return parsed content."""
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise, fact-grounded scriptwriter."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.LLM_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    if json_mode:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content
