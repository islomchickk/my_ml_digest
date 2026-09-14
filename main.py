"""
Digest: сбор статей → фильтрация LLM → отправка в Telegram.

Использование:
    uv run python main.py                    # провайдер из .env
    uv run python main.py --llm openai       # OpenAI
    uv run python main.py --llm gemini       # Gemini
    uv run python main.py --llm openrouter   # OpenRouter
    uv run python main.py --llm neuraldeep   # NeuralDeep
    uv run python main.py --no-stats         # без статистики Хабра
    uv run python main.py --dry-run          # без отправки в Telegram
    uv run python main.py --test-send        # отправить digest_output.json в TG_CHAT_ID
    uv run python main.py --bot              # постоянно обслуживать кнопки Telegram
    uv run python main.py --remember-sent    # импортировать уже отправленный дайджест в историю
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
from digest.llm.errors import LLMResponseError
from digest.llm.prompt import parse_llm_response
from digest.llm.selection import select_in_stages
from digest.bot import send_digest, run_bot
from digest.history import DigestStore, article_key
from digest.models import Article, ArticleStats, DigestEntry

ARTICLES_FILE = Path("articles.json")


DIGEST_OUTPUT_FILE = Path("digest_output.json")
LLM_RESPONSE_FILE = Path("llm_response.txt")
LLM_API_RESPONSE_FILE = Path("llm_response.json")


def _save_digest(entries: list[DigestEntry], mentions: list[DigestEntry]) -> None:
    output = {
        "top": [asdict(entry) for entry in entries],
        "honorable_mentions": [asdict(mention) for mention in mentions],
    }
    DIGEST_OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _load_saved_digest() -> tuple[list[DigestEntry], list[DigestEntry]]:
    text = DIGEST_OUTPUT_FILE.read_text(encoding="utf-8")
    entries, mentions = parse_llm_response(text, [])
    # Older files contain only the top array. Recover the missing section only
    # when the saved LLM answer describes exactly the same top entries.
    if isinstance(json.loads(text), list) and LLM_RESPONSE_FILE.exists():
        try:
            original_entries, original_mentions = parse_llm_response(
                LLM_RESPONSE_FILE.read_text(encoding="utf-8"), [],
            )
        except ValueError:
            print("Saved LLM answer is invalid; using the legacy digest without mentions")
        else:
            if entries == original_entries:
                mentions = original_mentions
                _save_digest(entries, mentions)
                print("Recovered honorable mentions and updated digest_output.json")
    return entries, mentions


def _test_send(config: Config) -> None:
    """Загружает digest_output.json и отправляет дайджест только в TG_CHAT_ID."""
    if not config.tg_bot_token:
        print("TG_BOT_TOKEN not set"); sys.exit(1)
    if not config.tg_chat_id:
        print("TG_CHAT_ID not set"); sys.exit(1)
    if not DIGEST_OUTPUT_FILE.exists():
        print("digest_output.json not found. Run the full pipeline first."); sys.exit(1)

    entries, mentions = _load_saved_digest()

    print(f"Loaded {len(entries)} entries + {len(mentions)} honorable mentions from digest_output.json")
    print(f"Sending digest to TG_CHAT_ID={config.tg_chat_id}...")
    asyncio.run(send_digest(entries, config.tg_bot_token, [config.tg_chat_id], mentions))
    print("Done!")


def generate_digest(
    config: Config, no_parse: bool = False, exclude_urls: set[str] | None = None,
) -> tuple[list[DigestEntry], list[DigestEntry]]:
    # 1. Collect articles
    if no_parse:
        print("=== Loading articles from articles.json ===")
        if not ARTICLES_FILE.exists():
            print("articles.json not found, nothing to process.")
            return [], []
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
        articles = collect_articles(
            fetch_stats=config.fetch_habr_stats, habr_stats_limit=config.habr_stats_limit,
            exclude_urls=exclude_urls,
        )

        # Save articles to articles.json
        articles_data = [asdict(a) for a in articles]
        with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(articles)} articles to articles.json")

    articles = [article for article in articles if article_key(article.url) not in (exclude_urls or set())]
    if not articles:
        print("No new articles available")
        return [], []

    # 2. LLM filter + summarize
    print(f"\n=== Filtering with LLM ({config.llm_provider}) ===")
    provider = get_provider(config)
    # Preserve each stage for diagnosis and keep the legacy paths pointing to
    # the latest response while generation is in progress.
    for path in (LLM_RESPONSE_FILE, LLM_API_RESPONSE_FILE):
        path.unlink(missing_ok=True)
        for stage in range(1, 4):
            path.with_name(f"{path.stem}_stage_{stage}{path.suffix}").unlink(missing_ok=True)

    def complete_stage(stage, system_prompt, user_prompt, schema):
        stage_text = LLM_RESPONSE_FILE.with_name(f"{LLM_RESPONSE_FILE.stem}_stage_{stage}{LLM_RESPONSE_FILE.suffix}")
        stage_api = LLM_API_RESPONSE_FILE.with_name(f"{LLM_API_RESPONSE_FILE.stem}_stage_{stage}{LLM_API_RESPONSE_FILE.suffix}")
        try:
            response = provider.complete(system_prompt, user_prompt, json_schema=schema)
        except LLMResponseError as error:
            LLM_API_RESPONSE_FILE.write_text(error.raw_response, encoding="utf-8")
            stage_api.write_text(error.raw_response, encoding="utf-8")
            print(f"LLM stage {stage}/3 error: {error}. Raw API response saved to {stage_api}")
            raise
        LLM_RESPONSE_FILE.write_text(response, encoding="utf-8")
        stage_text.write_text(response, encoding="utf-8")
        api_response = getattr(provider, "last_response_json", None)
        if api_response:
            LLM_API_RESPONSE_FILE.write_text(api_response, encoding="utf-8")
            stage_api.write_text(api_response, encoding="utf-8")
        return response

    entries, mentions = select_in_stages(articles, complete_stage)
    # The complete aggregate stays compatible with the saved-response recovery.
    LLM_RESPONSE_FILE.write_text(json.dumps({
        "top": [asdict(entry) for entry in entries],
        "honorable_mentions": [asdict(mention) for mention in mentions],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
    _save_digest(entries, mentions)
    print("Saved to digest_output.json")

    return entries, mentions


def main():
    parser = argparse.ArgumentParser(description="Weekly article digest")
    parser.add_argument("--llm", type=str, help="LLM provider")
    parser.add_argument("--no-stats", action="store_true", help="Skip fetching Habr stats")
    parser.add_argument("--dry-run", action="store_true", help="Don't send to Telegram")
    parser.add_argument("--no-parse", action="store_true", help="Load articles.json instead of RSS")
    parser.add_argument("--test-send", action="store_true", help="Send saved digest to TG_CHAT_ID")
    parser.add_argument("--bot", action="store_true", help="Listen for Telegram buttons continuously")
    parser.add_argument("--remember-sent", action="store_true", help="Import the saved digest into TG_CHAT_ID delivery history")
    args = parser.parse_args()
    config = Config.from_env()
    if args.llm:
        config.llm_provider = args.llm
    if args.no_stats:
        config.fetch_habr_stats = False
    try:
        if args.test_send:
            _test_send(config)
            return
        store = DigestStore()
        if args.remember_sent:
            if not config.tg_chat_id:
                raise ValueError("TG_CHAT_ID not set")
            entries, mentions = _load_saved_digest()
            store.remember(config.tg_chat_id, entries + mentions)
            print(f"Remembered {len(entries) + len(mentions)} previously sent articles")
            return
        if args.bot:
            asyncio.run(run_bot(config, lambda excluded: generate_digest(config, exclude_urls=excluded), store))
            return
        with store.generation_lock():
            history_chat = config.tg_chat_id or config.tg_channel_id
            excluded = store.sent_urls(history_chat) if history_chat else set()
            entries, mentions = generate_digest(config, args.no_parse, excluded)
            if not entries:
                return
            if args.dry_run:
                print("\n--dry-run: skipping Telegram send")
                return
            chat_ids = [chat for chat in (config.tg_chat_id, config.tg_channel_id) if chat]
            if not chat_ids or not config.tg_bot_token:
                print("Telegram credentials not configured, skipping send")
                return
            print(f"\nSending digest to {len(chat_ids)} chat(s)...")
            asyncio.run(send_digest(entries, config.tg_bot_token, chat_ids, mentions))
            print("Done!")
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Digest error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
