"""
llm_utils.py — Ollama wrapper + JSON extraction + Pydantic validation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

MODEL = "qwen2.5:7b"

T = TypeVar("T", bound=BaseModel)


def call_llm(system_prompt: str, user_prompt: str, label: str = "") -> str:
    """Send a chat completion to the local Ollama model and return raw text."""
    tag = f"[{label}] " if label else ""
    print(f"  {tag}calling LLM...")
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


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


def parse_and_validate(
    raw_text: str,
    schema: Type[T],
    label: str = "",
) -> T:
    """
    Extract JSON from raw LLM text and validate it against a Pydantic schema.

    Validation errors are printed with detail so issues are visible.
    Falls back to a default instance rather than crashing the pipeline,
    ensuring the orchestrator can always continue.
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

        # Best-effort: keep only fields that exist on the model
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
