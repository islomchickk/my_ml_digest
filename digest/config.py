import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Telegram
    tg_bot_token: str = ""
    tg_chat_id: str = ""       # личка
    tg_channel_id: str = ""    # канал (опционально)

    # LLM провайдер: claude | openai | gemini | openrouter | neuraldeep
    llm_provider: str = "neuraldeep"
    llm_model: str = ""  # если пусто — используется дефолт провайдера

    # API ключи
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    neuraldeep_api_key: str = ""

    # Парсер
    fetch_habr_stats: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            tg_bot_token=os.getenv("TG_BOT_TOKEN", ""),
            tg_chat_id=os.getenv("TG_CHAT_ID", ""),
            tg_channel_id=os.getenv("TG_CHANNEL_ID", ""),
            llm_provider=os.getenv("LLM_PROVIDER", "neuraldeep"),
            llm_model=os.getenv("LLM_MODEL", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            neuraldeep_api_key=os.getenv("NEURALDEEP_API_KEY", ""),
            fetch_habr_stats=os.getenv("FETCH_HABR_STATS", "true").lower() == "true",
        )
