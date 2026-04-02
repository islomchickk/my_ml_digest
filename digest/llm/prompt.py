"""Промпт для LLM: фильтрация + саммаризация статей."""

import json

from digest.models import Article, DigestEntry

SYSTEM_PROMPT = """\
Ты — AI-ассистент, который помогает составить еженедельный дайджест статей по Data Science, ML, AI и разработке.

Тебе дан JSON-массив статей из разных источников (Хабр, Medium, TDS).
Каждая статья содержит: title, preview, tags, source, и (для Хабра) stats (score, comments, views, bookmarks).

Твоя задача:
1. Отобрать 10 самых интересных и полезных статей.
2. Для каждой написать краткое саммари на русском языке (2-3 предложения).
3. Для каждой указать category — тему МО и подтему в скобках.
   Примеры: "LLM (агенты)", "NLP (embeddings)", "CV (сегментация)", "Speech (TTS)", "Classic ML (feature engineering)".
   Список тем неисчерпывающий — можешь предложить свою, если статья не вписывается в стандартные.
4. Дополнительно выбрать 3-4 статьи, которые не вошли в топ-10, но тоже могут быть интересны.
   Для них написать одно короткое предложение — о чём статья.

Приоритет тематик (от высшего к низшему):
1. LLM — агенты, skills, MCP, RAG, fine-tuning, prompt engineering, инференс
2. NLP — обработка текста, embeddings, классификация, NER, суммаризация
3. CV — компьютерное зрение, детекция, сегментация, генерация изображений
4. Speech — распознавание и синтез речи, аудио-модели
5. Classic ML — табличные данные, sklearn, XGBoost, feature engineering

Критерии отбора:
- Практическая польза (туториалы, гайды, best practices, код и примеры)
- Глубина (исследования, разборы архитектур, сравнения подходов)
- Для Хабра: высокий score, много комментариев и закладок — сигнал качества
- Разнообразие источников (не все 10 из одного источника)

НЕ интересны:
- Новостные заметки без глубины (новости читатель получает из других каналов)
- Рекламные / промо статьи
- Слишком вводные статьи ("что такое ML", "введение в нейросети")
- Подборки инструментов без анализа и сравнения
"""

_ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "url": {"type": "string"},
        "source": {"type": "string"},
        "author": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "description": "Краткое саммари на русском, 2-3 предложения"},
        "category": {"type": "string", "description": "Тема МО и подтема в скобках, например: LLM (RAG), CV (детекция), NLP (NER)"},
    },
    "required": ["title", "url", "source", "author", "tags", "summary", "category"],
    "additionalProperties": False,
}

_MENTION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "url": {"type": "string"},
        "source": {"type": "string"},
        "summary": {"type": "string", "description": "Одно предложение — о чём статья"},
    },
    "required": ["title", "url", "source", "summary"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "top": {"type": "array", "items": _ARTICLE_SCHEMA},
        "honorable_mentions": {"type": "array", "items": _MENTION_SCHEMA},
    },
    "required": ["top", "honorable_mentions"],
    "additionalProperties": False,
}


def build_user_prompt(articles: list[Article]) -> str:
    items = []
    for art in articles:
        entry: dict = {
            "title": art.title,
            "url": art.url,
            "source": art.source,
            "author": art.author,
            "tags": art.tags[:5],
            "preview": art.preview[:400],
        }
        if art.source == "habr":
            entry["stats"] = {
                "score": art.stats.score,
                "comments": art.stats.comments,
                "views": art.stats.views,
                "bookmarks": art.stats.bookmarks,
            }
        items.append(entry)
    return json.dumps(items, ensure_ascii=False, indent=2)


def _parse_entries(items: list[dict], url_to_article: dict[str, Article]) -> list[DigestEntry]:
    entries = []
    for item in items:
        url = item.get("url", "")
        original = url_to_article.get(url)
        entries.append(DigestEntry(
            title=item.get("title", ""),
            url=url,
            source=item.get("source", ""),
            author=item.get("author", ""),
            tags=item.get("tags", []),
            summary=item.get("summary", ""),
            category=item.get("category", ""),
            stats=original.stats if original else None,
        ))
    return entries


def parse_llm_response(
    response: str, articles: list[Article],
) -> tuple[list[DigestEntry], list[DigestEntry]]:
    """Парсим JSON-ответ LLM. Возвращает (top_10, honorable_mentions)."""
    data = json.loads(response.strip())
    url_to_article = {a.url: a for a in articles}

    # New format: {"top": [...], "honorable_mentions": [...]}
    if isinstance(data, dict) and "top" in data:
        top = _parse_entries(data["top"], url_to_article)
        mentions = _parse_entries(data.get("honorable_mentions", []), url_to_article)
        return top, mentions

    # Legacy: bare array or {"articles": [...]}
    if isinstance(data, dict):
        data = data.get("articles", [])
    return _parse_entries(data, url_to_article), []
