import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_HABR_STATS_LIMIT = 50
DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS = 8_000
DEFAULT_NEURALDEEP_THINKING_TOKEN_BUDGET = 1_024


@dataclass
class Config:
    # Telegram
    tg_bot_token: str = ""
    tg_chat_id: str = ""       # личка
    tg_channel_id: str = ""    # канал (опционально)

    # LLM провайдер: claude | openai | gemini | openrouter | neuraldeep
    llm_provider: str = "neuraldeep"
    llm_model: str = ""  # если пусто — используется дефолт провайдера
    neuraldeep_max_output_tokens: int = DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS
    neuraldeep_enable_thinking: bool = False
    neuraldeep_thinking_token_budget: int = DEFAULT_NEURALDEEP_THINKING_TOKEN_BUDGET

    # API ключи
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    neuraldeep_api_key: str = ""

    # Парсер
    fetch_habr_stats: bool = True
    habr_stats_limit: int = DEFAULT_HABR_STATS_LIMIT

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            tg_bot_token=os.getenv("TG_BOT_TOKEN", ""),
            tg_chat_id=os.getenv("TG_CHAT_ID", ""),
            tg_channel_id=os.getenv("TG_CHANNEL_ID", ""),
            llm_provider=os.getenv("LLM_PROVIDER", "neuraldeep"),
            llm_model=os.getenv("LLM_MODEL", ""),
            neuraldeep_max_output_tokens=int(os.getenv(
                "NEURALDEEP_MAX_OUTPUT_TOKENS", str(DEFAULT_NEURALDEEP_MAX_OUTPUT_TOKENS),
            )),
            neuraldeep_enable_thinking=_env_bool("NEURALDEEP_ENABLE_THINKING", False),
            neuraldeep_thinking_token_budget=int(os.getenv(
                "NEURALDEEP_THINKING_TOKEN_BUDGET", str(DEFAULT_NEURALDEEP_THINKING_TOKEN_BUDGET),
            )),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            neuraldeep_api_key=os.getenv("NEURALDEEP_API_KEY", ""),
            fetch_habr_stats=os.getenv("FETCH_HABR_STATS", "true").lower() == "true",
            habr_stats_limit=int(os.getenv("HABR_STATS_LIMIT", str(DEFAULT_HABR_STATS_LIMIT))),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"
