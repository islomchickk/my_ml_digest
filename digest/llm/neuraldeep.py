import re
from typing import Any

import openai

from digest.config import (
    DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS,
    DEFAULT_NEURALDEEP_MODEL_NO_THINKING,
    DEFAULT_NEURALDEEP_MODEL_THINKING,
    DEFAULT_NEURALDEEP_THINKING_TOKEN_BUDGET,
)
from digest.llm.openai import OpenAIProvider
from digest.llm.token_budget import fit_article_prompt, request_overhead

BASE_URL = "https://api.neuraldeep.ru/v1"
MAX_INPUT_TOKENS = 46_112
TOKEN_SAFETY_MARGIN = 1_024
MAX_ATTEMPTS = 3

_INPUT_LIMIT_ERROR = re.compile(r"(\d+)\s+токенов входа при пределе\s+(\d+)")


class NeuralDeepProvider(OpenAIProvider):
    """NeuralDeep через OpenAI-совместимый Chat Completions API."""

    def __init__(
        self, api_key: str, model: str = "",
        max_output_tokens: int = DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS,
        enable_thinking: bool = False,
        thinking_token_budget: int = DEFAULT_NEURALDEEP_THINKING_TOKEN_BUDGET,
    ):
        if not api_key:
            raise ValueError("NEURALDEEP_API_KEY is not set")
        if not 1 <= max_output_tokens <= DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS:
            raise ValueError("NEURALDEEP_MAX_OUTPUT_TOKENS must be between 1 and 8000")
        if enable_thinking and not 1 <= thinking_token_budget < max_output_tokens:
            raise ValueError(
                "NEURALDEEP_THINKING_TOKEN_BUDGET must be positive and less than "
                "NEURALDEEP_MAX_OUTPUT_TOKENS when thinking is enabled"
            )
        self.enable_thinking = enable_thinking
        self.thinking_token_budget = thinking_token_budget
        template_kwargs = {"enable_thinking": enable_thinking}
        if enable_thinking:
            template_kwargs["thinking_token_budget"] = thinking_token_budget
        super().__init__(
            api_key=api_key,
            model=model or (DEFAULT_NEURALDEEP_MODEL_THINKING if enable_thinking
                            else DEFAULT_NEURALDEEP_MODEL_NO_THINKING),
            base_url=BASE_URL,
            max_output_tokens=max_output_tokens,
            extra_body={"chat_template_kwargs": template_kwargs},
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        overhead = request_overhead(system_prompt, json_schema)
        budget = MAX_INPUT_TOKENS - TOKEN_SAFETY_MARGIN
        for attempt in range(MAX_ATTEMPTS):
            prepared = fit_article_prompt(user_prompt, overhead, budget)
            print(
                f"NeuralDeep: model={self.model}; {prepared.article_count} articles; estimated input "
                f"{prepared.estimated_tokens} tokens (budget {budget}, "
                f"plan limit {MAX_INPUT_TOKENS}; tokenizer estimate); "
                f"max output {self.max_output_tokens} tokens; "
                f"thinking={'on' if self.enable_thinking else 'off'}"
                + (f" (budget {self.thinking_token_budget} tokens)" if self.enable_thinking else "")
            )
            try:
                return super().complete(system_prompt, prepared.text, json_schema)
            except openai.BadRequestError as error:
                # Only retry the specific input-size rejection, not auth/schema
                # errors. Calibrate the local estimate with the server's count.
                match = _INPUT_LIMIT_ERROR.search(str(error))
                if match is None or attempt == MAX_ATTEMPTS - 1:
                    raise
                actual, limit = map(int, match.groups())
                if actual <= limit or limit <= TOKEN_SAFETY_MARGIN:
                    raise
                budget = min(
                    budget - 1,
                    int(prepared.estimated_tokens
                        * (min(limit, MAX_INPUT_TOKENS) - TOKEN_SAFETY_MARGIN)
                        / actual * 0.95),
                )
                print(
                    f"NeuralDeep reported {actual} input tokens, limit {limit}; "
                    f"reducing estimated budget to {budget} and retrying"
                )
        raise RuntimeError("NeuralDeep request attempts exhausted")
