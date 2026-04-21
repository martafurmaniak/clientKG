"""
llm_utils.py — LLM backend abstraction + JSON extraction + Pydantic validation.

Supports two backends, selected by the BACKEND constant below:
  "azure"  — Azure OpenAI (e.g. GPT-4.1 deployment)
  "ollama" — local Ollama model (e.g. qwen2.5:7b)

Azure configuration is read from environment variables so credentials
are never hard-coded:
    AZURE_OPENAI_ENDPOINT   e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_KEY    your API key
    AZURE_OPENAI_API_VERSION e.g. 2024-12-01-preview
    AZURE_OPENAI_DEPLOYMENT  your deployment name, e.g. gpt-4-1
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

# ══════════════════════════════════════════════════════════════
#  ▶  CONFIGURE HERE
# ══════════════════════════════════════════════════════════════

BACKEND = "azure"       # "azure" | "ollama"

# Azure settings — override via environment variables (recommended)
# or set directly here for quick testing
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT",    "https://<your-resource>.openai.azure.com/")
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY",     "<your-api-key>")
AZURE_API_VERSION= os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT",  "gpt-4-1")

# Token budget for each LLM call.
# GPT-4.1 supports up to 32 768 output tokens; 4 096 is a safe default
# that avoids runaway costs while being far larger than qwen2.5:7b's limit.
MAX_OUTPUT_TOKENS = 32768  # GPT-4.1 maximum output tokens

# Ollama settings (only used when BACKEND = "ollama")
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_CTX   = 16384   # context window — increase if you have the RAM

# ══════════════════════════════════════════════════════════════

T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────────────────────────────────────
# Backend initialisation (lazy — only imported when needed)
# ─────────────────────────────────────────────────────────────────────────────

def _get_azure_client():
    try:
        from openai import AzureOpenAI
    except ImportError as e:
        raise ImportError(
            "openai package not found. Install with: pip install openai"
        ) from e
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, label: str = "") -> str:
    """
    Send a chat completion and return the raw response text.

    Logs a clear warning when the response was truncated due to hitting
    the token limit (finish_reason / done_reason == "length"), because
    truncated JSON is the most common cause of downstream Pydantic errors.
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}calling LLM ({BACKEND})...")

    if BACKEND == "azure":
        return _call_azure(system_prompt, user_prompt, tag)
    elif BACKEND == "ollama":
        return _call_ollama(system_prompt, user_prompt, tag)
    else:
        raise ValueError(f"Unknown BACKEND '{BACKEND}'. Use 'azure' or 'ollama'.")


def _call_azure(system_prompt: str, user_prompt: str, tag: str) -> str:
    client = _get_azure_client()
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    choice       = response.choices[0]
    finish_reason = choice.finish_reason
    content      = choice.message.content or ""

    # Log token usage so you can monitor prompt size
    usage = response.usage
    if usage:
        print(f"  {tag}tokens — prompt={usage.prompt_tokens}  "
              f"completion={usage.completion_tokens}  total={usage.total_tokens}")

    # Truncation detection
    if finish_reason == "length":
        print(
            f"  {tag}⚠ RESPONSE TRUNCATED (finish_reason=length). "
            f"The model hit MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS}. "
            f"Increase MAX_OUTPUT_TOKENS or reduce prompt size."
        )
    else:
        print(f"  {tag}finish_reason={finish_reason}")

    return content


def _call_ollama(system_prompt: str, user_prompt: str, tag: str) -> str:
    import ollama as _ollama
    response = _ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        options={"temperature": 0.0, "num_ctx": OLLAMA_CTX},
    )

    done_reason = response.get("done_reason", "unknown")
    content     = response["message"]["content"]

    if done_reason == "length":
        print(
            f"  {tag}⚠ RESPONSE TRUNCATED (done_reason=length). "
            f"Increase OLLAMA_CTX (currently {OLLAMA_CTX}) or reduce prompt size."
        )
    else:
        print(f"  {tag}done_reason={done_reason}")

    return content


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | list:
    """
    Pull the first JSON object or array out of an LLM response.
    Handles markdown code fences and stray surrounding text.
    """
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)

    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not extract valid JSON from LLM response.\nRaw text:\n{text}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic validation
# ─────────────────────────────────────────────────────────────────────────────

def parse_and_validate(
    raw_text: str,
    schema: Type[T],
    label: str = "",
) -> T:
    """
    Extract JSON from raw LLM text and validate against a Pydantic schema.
    Logs detailed errors; falls back to a default instance so the pipeline
    never hard-crashes on a bad LLM response.
    """
    tag = f"[{label}] " if label else ""

    raw_data: Any = None
    try:
        raw_data = extract_json(raw_text)
    except ValueError as exc:
        print(f"  {tag}⚠ JSON extraction failed: {exc}")
        return schema()  # type: ignore[call-arg]

    try:
        return schema.model_validate(raw_data)
    except ValidationError as exc:
        print(f"  {tag}⚠ Pydantic validation errors:")
        for error in exc.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            print(f"      field={loc}  type={error['type']}  msg={error['msg']}")

        if isinstance(raw_data, dict):
            valid_fields = schema.model_fields.keys()
            filtered = {k: v for k, v in raw_data.items() if k in valid_fields}
            try:
                instance = schema.model_validate(filtered)
                print(f"  {tag}  Partial validation succeeded with filtered fields.")
                return instance
            except ValidationError:
                pass

        print(f"  {tag}⚠ Falling back to empty schema instance.")
        return schema()  # type: ignore[call-arg]
