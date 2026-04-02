"""
Digest: сбор статей → фильтрация LLM → отправка в Telegram.

Использование:
    uv run python main.py                    # Claude по умолчанию
    uv run python main.py --llm openai       # OpenAI
    uv run python main.py --llm gemini       # Gemini
    uv run python main.py --llm openrouter   # OpenRouter
    uv run python main.py --no-stats         # без статистики Хабра
    uv run python main.py --dry-run          # без отправки в Telegram
    uv run python main.py --test-send        # отправить digest_output.json в TG_CHAT_ID
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from digest.config import Config
from digest.parser import collect_articles
from digest.llm import get_provider
from digest.llm.prompt import SYSTEM_PROMPT, RESPONSE_SCHEMA, build_user_prompt, parse_llm_response
from digest.bot import send_digest
from digest.models import Article, ArticleStats, DigestEntry

ARTICLES_FILE = Path("articles.json")


DIGEST_OUTPUT_FILE = Path("digest_output.json")


def _test_send(config: Config) -> None:
    """Загружает digest_output.json и отправляет дайджест только в TG_CHAT_ID."""
    if not config.tg_bot_token:
        print("TG_BOT_TOKEN not set"); sys.exit(1)
    if not config.tg_chat_id:
        print("TG_CHAT_ID not set"); sys.exit(1)
    if not DIGEST_OUTPUT_FILE.exists():
        print("digest_output.json not found. Run the full pipeline first."); sys.exit(1)

    with open(DIGEST_OUTPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = [
        DigestEntry(
            title=e["title"],
            url=e["url"],
            source=e["source"],
            author=e["author"],
            tags=e.get("tags", []),
            summary=e["summary"],
            category=e.get("category", ""),
        )
        for e in raw
    ]

    print(f"Loaded {len(entries)} entries from digest_output.json")
    print(f"Sending digest to TG_CHAT_ID={config.tg_chat_id}...")
    asyncio.run(send_digest(entries, config.tg_bot_token, [config.tg_chat_id]))
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Weekly article digest")
    parser.add_argument("--llm", type=str, help="LLM provider: claude, openai, gemini, openrouter")
    parser.add_argument("--no-stats", action="store_true", help="Skip fetching Habr stats")
    parser.add_argument("--dry-run", action="store_true", help="Don't send to Telegram")
    parser.add_argument("--no-parse", action="store_true", help="Skip parsing, load from articles.json")
    parser.add_argument("--test-send", action="store_true", help="Send digest from digest_output.json to TG_CHAT_ID only (bypass parser & LLM)")
    args = parser.parse_args()

    config = Config.from_env()

    if args.test_send:
        _test_send(config)
        return
    if args.llm:
        config.llm_provider = args.llm
    if args.no_stats:
        config.fetch_habr_stats = False

    # 1. Collect articles
    if args.no_parse:
        print("=== Loading articles from articles.json ===")
        if not ARTICLES_FILE.exists():
            print("articles.json not found, nothing to process.")
            sys.exit(0)
        with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        articles = [
            Article(
                title=a["title"],
                url=a["url"],
                source=a["source"],
                author=a["author"],
                published=a["published"],
                tags=a.get("tags", []),
                preview=a.get("preview", ""),
                stats=ArticleStats(**a["stats"]) if a.get("stats") else ArticleStats(),
            )
            for a in raw
        ]
        print(f"Loaded {len(articles)} articles")
    else:
        print("=== Collecting articles ===")
        articles = collect_articles(fetch_stats=config.fetch_habr_stats)

        # Save articles to articles.json
        articles_data = [asdict(a) for a in articles]
        with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(articles)} articles to articles.json")

    # 2. LLM filter + summarize
    print(f"\n=== Filtering with LLM ({config.llm_provider}) ===")
    provider = get_provider(config)
    user_prompt = build_user_prompt(articles)

    print(f"Sending {len(articles)} articles to LLM...")
    response = provider.complete(SYSTEM_PROMPT, user_prompt, json_schema=RESPONSE_SCHEMA)

    entries, mentions = parse_llm_response(response, articles)
    print(f"LLM selected {len(entries)} articles + {len(mentions)} honorable mentions\n")

    for i, e in enumerate(entries, 1):
        print(f"  {i}. [{e.source}] {e.title}")
        print(f"     {e.summary[:100]}...")
        print()

    if mentions:
        print("  Также может быть интересно:")
        for m in mentions:
            print(f"  • [{m.source}] {m.title}")
            print(f"    {m.summary}")
            print()

    # 3. Save to JSON
    output = [
        {
            "title": e.title,
            "url": e.url,
            "source": e.source,
            "author": e.author,
            "tags": e.tags,
            "summary": e.summary,
            "category": e.category,
        }
        for e in entries
    ]
    with open("digest_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("Saved to digest_output.json")

    # 4. Send to Telegram
    if args.dry_run:
        print("\n--dry-run: skipping Telegram send")
        return

    chat_ids = []
    if config.tg_chat_id:
        chat_ids.append(config.tg_chat_id)
    if config.tg_channel_id:
        chat_ids.append(config.tg_channel_id)

    if not chat_ids:
        print("\nNo TG_CHAT_ID or TG_CHANNEL_ID set, skipping Telegram send")
        return

    if not config.tg_bot_token:
        print("\nNo TG_BOT_TOKEN set, skipping Telegram send")
        return

    print(f"\nSending digest to {len(chat_ids)} chat(s)...")
    asyncio.run(send_digest(entries, config.tg_bot_token, chat_ids, mentions))
    print("Done!")


if __name__ == "__main__":
    main()
