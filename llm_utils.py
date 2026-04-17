"""
llm_utils.py — thin wrapper around the Ollama Python client.
All agents use call_llm(); JSON extraction is centralised here.
"""

import json
import re
import ollama

MODEL = "qwen2.5:7b"


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
    # Strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)

    # Find the outermost { } or [ ]
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

    # Last resort: try parsing the whole string
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not extract valid JSON from LLM response.\nRaw text:\n{text}") from exc
