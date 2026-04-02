from abc import ABC, abstractmethod
from typing import Any

from digest.config import Config


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """If json_schema is provided, the response is guaranteed valid JSON."""
        ...


def get_provider(config: Config) -> LLMProvider:
    name = config.llm_provider.lower()

    if name == "claude":
        from digest.llm.claude import ClaudeProvider
        return ClaudeProvider(config.anthropic_api_key, config.llm_model)
    elif name == "openai":
        from digest.llm.openai import OpenAIProvider
        return OpenAIProvider(config.openai_api_key, config.llm_model)
    elif name == "gemini":
        from digest.llm.gemini import GeminiProvider
        return GeminiProvider(config.gemini_api_key, config.llm_model)
    elif name == "openrouter":
        from digest.llm.openrouter import OpenRouterProvider
        return OpenRouterProvider(config.openrouter_api_key, config.llm_model)
    else:
        raise ValueError(f"Unknown LLM provider: {name}")
