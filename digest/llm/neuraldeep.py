from digest.llm.openai import OpenAIProvider

DEFAULT_MODEL = "qwen3.6-unlim"
BASE_URL = "https://api.neuraldeep.ru/v1"


class NeuralDeepProvider(OpenAIProvider):
    """NeuralDeep через OpenAI-совместимый Chat Completions API."""

    def __init__(self, api_key: str, model: str = ""):
        if not api_key:
            raise ValueError("NEURALDEEP_API_KEY is not set")
        super().__init__(
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            base_url=BASE_URL,
        )
