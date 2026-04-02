from typing import Any

import openai

from digest.llm.base import LLMProvider

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "digest_response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
