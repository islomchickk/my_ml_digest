from typing import Any

import httpx

from digest.llm.base import LLMProvider

DEFAULT_MODEL = "anthropic/claude-sonnet-4"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

JSON_INSTRUCTION = "\n\nОтветь строго в формате JSON-массива, без markdown-обёртки."


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        system = system_prompt
        if json_schema:
            system += JSON_INSTRUCTION
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"] or ""
        # Strip markdown wrapper if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return text
