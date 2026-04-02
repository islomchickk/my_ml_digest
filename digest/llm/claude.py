from typing import Any

import anthropic

from digest.llm.base import LLMProvider

DEFAULT_MODEL = "claude-sonnet-4-20250514"

JSON_INSTRUCTION = "\n\nОтветь строго в формате JSON-массива, без markdown-обёртки."


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.client = anthropic.Anthropic(api_key=api_key)
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
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text
        # Strip markdown wrapper if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return text
