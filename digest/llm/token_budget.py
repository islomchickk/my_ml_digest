"""Local request-size estimates and JSON-safe article prompt trimming."""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # Qwen's gateway alias has no published tokenizer mapping. This is an
    # estimate; NeuralDeep's reported input count remains authoritative.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def request_overhead(system_prompt: str, json_schema: dict[str, Any] | None) -> int:
    tokens = count_tokens(system_prompt) + 16  # message roles and delimiters
    if json_schema:
        tokens += count_tokens(json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": "digest_response", "schema": json_schema, "strict": True,
            },
        }, ensure_ascii=False))
    return tokens


@dataclass
class PreparedPrompt:
    text: str
    estimated_tokens: int
    article_count: int


def fit_article_prompt(user_prompt: str, overhead: int, budget: int) -> PreparedPrompt:
    """Shorten previews first, then omit trailing articles if metadata won't fit.

    Never slice serialized JSON, article URLs, or the system/schema instructions.
    """
    items = json.loads(user_prompt)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("Expected a JSON array of articles for token budgeting")

    def prepare(articles: list[dict]) -> PreparedPrompt:
        text = json.dumps(articles, ensure_ascii=False, separators=(",", ":"))
        return PreparedPrompt(text, overhead + count_tokens(text), len(articles))

    full = prepare(items)
    if full.estimated_tokens <= budget:
        return full

    def with_previews(length: int) -> PreparedPrompt:
        return prepare([
            {**item, "preview": item.get("preview", "")[:length]} for item in items
        ])

    best = with_previews(0)
    if best.estimated_tokens <= budget:
        low, high = 0, max(len(item.get("preview", "")) for item in items)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = with_previews(middle)
            if candidate.estimated_tokens <= budget:
                low, best = middle, candidate
            else:
                high = middle - 1
        return best

    # Metadata alone exceeds the budget. Keep complete records in source order.
    minimal_items = json.loads(best.text)
    best = prepare([])
    low, high = 0, len(items)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = prepare(minimal_items[:middle])
        if candidate.estimated_tokens <= budget:
            low, best = middle, candidate
        else:
            high = middle - 1
    if not best.article_count:
        raise ValueError("Token budget is too small for the instructions, schema and one article")
    return best
