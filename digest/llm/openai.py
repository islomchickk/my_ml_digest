from copy import deepcopy
from typing import Any

import openai

from digest.llm.base import LLMProvider
from digest.llm.errors import LLMResponseError

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    def __init__(
        self, api_key: str, model: str = "", base_url: str | None = None,
        max_output_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or DEFAULT_MODEL
        self.max_output_tokens = max_output_tokens
        self.extra_body = deepcopy(extra_body)

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
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.extra_body is not None:
            kwargs["extra_body"] = deepcopy(self.extra_body)
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
        self.last_response_json = response.model_dump_json(indent=2)
        if not response.choices:
            raise LLMResponseError("LLM returned no choices", self.last_response_json)
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        reasoning = getattr(choice.message, "reasoning_content", None)
        print(
            f"LLM response: finish_reason={choice.finish_reason}, "
            f"input_tokens={usage.prompt_tokens if usage else 'unknown'}, "
            f"output_tokens={usage.completion_tokens if usage else 'unknown'}, "
            f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'unknown'}, "
            f"reasoning_chars={len(reasoning) if isinstance(reasoning, str) else 0}, "
            f"answer_chars={len(choice.message.content or '')}"
        )
        if choice.finish_reason == "length":
            raise LLMResponseError(
                "LLM output was truncated (finish_reason=length); "
                "the output token limit was reached before completion",
                self.last_response_json,
            )
        if choice.message.refusal:
            raise LLMResponseError("LLM refused the request", self.last_response_json)
        if choice.finish_reason != "stop":
            raise LLMResponseError(
                f"LLM returned an incomplete answer (finish_reason={choice.finish_reason})",
                self.last_response_json,
            )
        content = choice.message.content
        if not content or not content.strip():
            raise LLMResponseError(
                "LLM returned empty message.content; inspect the saved API response "
                "for reasoning_content and token usage",
                self.last_response_json,
            )
        return content
