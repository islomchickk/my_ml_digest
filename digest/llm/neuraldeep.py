import re
from typing import Any

import openai

from digest.config import DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS
from digest.llm.openai import OpenAIProvider
from digest.llm.token_budget import fit_article_prompt, request_overhead

DEFAULT_MODEL = "qwen3.6-unlim"
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
    ):
        if not api_key:
            raise ValueError("NEURALDEEP_API_KEY is not set")
        if not 1 <= max_output_tokens <= DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS:
            raise ValueError("NEURALDEEP_MAX_OUTPUT_TOKENS must be between 1 and 8000")
        super().__init__(
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            base_url=BASE_URL,
            max_output_tokens=max_output_tokens,
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
                f"NeuralDeep: {prepared.article_count} articles; estimated input "
                f"{prepared.estimated_tokens} tokens (budget {budget}, "
                f"plan limit {MAX_INPUT_TOKENS}; tokenizer estimate); "
                f"max output {self.max_output_tokens} tokens"
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
