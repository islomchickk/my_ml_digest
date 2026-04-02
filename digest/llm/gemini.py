from typing import Any

from google import genai

from digest.llm.base import LLMProvider

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        config = genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
        if json_schema:
            config.response_mime_type = "application/json"
            config.response_schema = json_schema
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return response.text or ""
