"""
Сборка статей из RSS-фидов Хабра, Medium, TDS.
Статистика Хабра извлекается из встроенного JSON (window.__PINIA_STATE__).
"""

import feedparser
import httpx
import json
import re
import html
import time

from digest.models import Article, ArticleStats


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

FEEDS = {
    "habr": [
        ("ML",       "https://habr.com/ru/rss/hubs/machine_learning/articles/top/weekly/?fl=ru"),
        ("AI",       "https://habr.com/ru/rss/hubs/artificial_intelligence/articles/top/weekly/?fl=ru"),
        ("NLP",      "https://habr.com/ru/rss/hubs/natural_language_processing/articles/top/weekly/?fl=ru"),
        ("DataEng",  "https://habr.com/ru/rss/hubs/data_engineering/articles/top/weekly/?fl=ru"),
        ("DataSci",  "https://habr.com/ru/rss/hubs/data_science/articles/top/weekly/?fl=ru"),
        ("Python",   "https://habr.com/ru/rss/hubs/python/articles/top/weekly/?fl=ru"),
    ],
    "medium": [
        ("ML",          "https://medium.com/feed/tag/machine-learning"),
        ("LLM",         "https://medium.com/feed/tag/llm"),
        ("RAG",         "https://medium.com/feed/tag/rag"),
        ("AI",          "https://medium.com/feed/tag/artificial-intelligence"),
        ("MLOps",       "https://medium.com/feed/tag/mlops"),
        ("DataScience", "https://medium.com/feed/tag/data-science"),
    ],
    "tds": [
        ("TDS", "https://towardsdatascience.com/feed"),
    ],
}


def clean_html(raw_html: str, max_chars: int = 600) -> str:
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] + "..." if len(text) > max_chars else text


def extract_article_id(url: str) -> str | None:
    match = re.search(r"/articles/(\d+)", url)
    return match.group(1) if match else None


def fetch_habr_stats(url: str, client: httpx.Client) -> ArticleStats | None:
    article_id = extract_article_id(url)
    if not article_id:
        return None

    try:
        resp = client.get(url, follow_redirects=True, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [habr] error fetching {url}: {e}")
        return None

    match = re.search(
        r"window\.__PINIA_STATE__\s*=\s*(\{.+?\});\s*\(function",
        resp.text,
        re.DOTALL,
    )
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    articles_list = data.get("articlesList", {}).get("articlesList", {})
    article_data = articles_list.get(article_id, {})
    stats_raw = article_data.get("statistics", {})

    if not stats_raw:
        return None

    return ArticleStats(
        score=stats_raw.get("score", 0),
        votes_plus=stats_raw.get("votesCountPlus", 0),
        votes_minus=stats_raw.get("votesCountMinus", 0),
        comments=stats_raw.get("commentsCount", 0),
        bookmarks=stats_raw.get("favoritesCount", 0),
        views=stats_raw.get("readingCount", 0),
        reach=stats_raw.get("reach", 0),
    )


def parse_feed(url: str, source: str) -> list[Article]:
    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries:
        tags = [t.get("term", "") for t in getattr(entry, "tags", []) if t.get("term")]
        preview = clean_html(entry.get("summary", entry.get("description", "")))
        published = getattr(entry, "published", "")

        articles.append(Article(
            title=entry.get("title", "").strip(),
            url=entry.get("link", ""),
            source=source,
            author=entry.get("author", "—"),
            published=published,
            tags=tags,
            preview=preview,
        ))

    return articles


def collect_articles(fetch_stats: bool = True) -> list[Article]:
    """Собирает все статьи из RSS и обогащает статистикой Хабра."""
    all_articles: list[Article] = []
    seen_urls: set[str] = set()

    for source, feeds in FEEDS.items():
        for label, url in feeds:
            print(f"[{source}] {label}")
            try:
                articles = parse_feed(url, source)
            except Exception as e:
                print(f"  error: {e}")
                continue

            new_count = 0
            for art in articles:
                if art.url not in seen_urls:
                    seen_urls.add(art.url)
                    all_articles.append(art)
                    new_count += 1

            print(f"  {len(articles)} fetched, {new_count} new")

    habr_articles = [a for a in all_articles if a.source == "habr"]

    if fetch_stats and habr_articles:
        print(f"\nFetching stats for {len(habr_articles)} Habr articles...")
        with httpx.Client(headers=HEADERS) as client:
            for i, art in enumerate(habr_articles, 1):
                stats = fetch_habr_stats(art.url, client)
                if stats:
                    art.stats = stats
                    print(f"  [{i}/{len(habr_articles)}] {art.title[:50]}  "
                          f"[+{stats.score} c:{stats.comments} v:{stats.views}]")
                else:
                    print(f"  [{i}/{len(habr_articles)}] no stats: {art.title[:50]}")
                time.sleep(0.5)

    print(f"\nTotal: {len(all_articles)} unique articles")
    return all_articles
