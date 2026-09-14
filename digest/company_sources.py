"""Official company blogs and strict seven-day publication filtering."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from digest.models import Article

COMPANY_FEEDS = {
    "openai": [("Developer blog", "https://developers.openai.com/blog/")],
    "anthropic": [
        ("Engineering", "https://www.anthropic.com/engineering"),
        ("Research", "https://www.anthropic.com/research"),
        ("News", "https://www.anthropic.com/news"),
    ],
    "deepmind": [("Research and models", "https://deepmind.google/blog/rss.xml")],
    "meta": [("AI research", "https://ai.meta.com/blog/")],
    "mistral": [("Research and engineering", "https://mistral.ai/news/rss")],
    "huggingface": [("Models and tooling", "https://huggingface.co/blog/feed.xml")],
}
HTML_BLOGS = {
    "https://developers.openai.com/blog/": ("a.resource-item[href]", "/blog/", "OpenAI"),
    "https://www.anthropic.com/engineering": ("article a[href]", "/engineering/", "Anthropic"),
    "https://www.anthropic.com/research": ("article a[href], a[class*='PublicationList']", "/research/", "Anthropic"),
    "https://www.anthropic.com/news": ("a[class*='FeaturedGrid'], a[class*='PublicationList']", "/news/", "Anthropic"),
    "https://ai.meta.com/blog/": ("a._amdf[href], a._amd2[href]", "/blog/", "Meta"),
}
_DATE = re.compile(
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}", re.I,
)
_MONTH_DAY = re.compile(r"^[A-Za-z]+\s+\d{1,2}$")


def publication_date(value: str) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    for parse in (lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            result = parse(value)
            return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            pass
    match = _DATE.fullmatch(value)
    if match:
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def is_recent_publication(value: str, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    published = publication_date(value)
    return published is not None and now - timedelta(days=7) <= published <= now


def filter_company_articles(articles: list[Article], now: datetime | None = None) -> list[Article]:
    now = now or datetime.now(timezone.utc)
    return [a for a in articles if a.source not in COMPANY_FEEDS or is_recent_publication(a.published, now)]


def _visible_date(node, *, publication_header: bool = False) -> str:
    selector = ("time, [class*='date'], span.text-default.font-medium" if publication_header
                else "time, span, p, div")
    for element in node.select(selector):
        text = element.get_text(" ", strip=True)
        classes = " ".join(element.get("class", [])).lower()
        if "updated" in classes or text.lower().startswith("updated"):
            continue
        value = element.get("datetime", "") if element.name == "time" else ""
        if publication_date(value):
            return value
        text = re.sub(r"^Published\s+", "", text, flags=re.I)
        if publication_date(text):
            return text
    return ""


def _json_articles(value):
    if isinstance(value, list):
        for item in value:
            yield from _json_articles(item)
    elif isinstance(value, dict):
        types = value.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if any(t in {"Article", "BlogPosting", "NewsArticle", "TechArticle"} for t in types):
            yield value
        yield from _json_articles(value.get("@graph", []))


def _fetch_article(
    client, url: str, source: str, company: str, card_preview: str, now: datetime,
    card_published: str = "",
) -> Article | None:
    try:
        response = client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.select_one("main") or soup
        structured = {}
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                structured = next(iter(_json_articles(json.loads(script.get_text()))), structured)
            except (ValueError, TypeError):
                continue
        published = structured.get("datePublished", "")
        if not publication_date(published):
            meta = soup.select_one('meta[property="article:published_time"]')
            published = meta.get("content", "") if meta else ""
        if not publication_date(published):
            published = _visible_date(main, publication_header=True) or card_published
        if not is_recent_publication(published, now):
            return None
        heading = main.select_one("h1")
        title = heading.get_text(" ", strip=True) if heading else structured.get("headline", "")
        if not title:
            return None
        preview = card_preview or structured.get("description", "")
        if not preview:
            paragraph = next((p for p in main.select("p") if len(p.get_text(strip=True)) > 80), None)
            preview = paragraph.get_text(" ", strip=True) if paragraph else ""
        if not preview:
            meta = soup.select_one('meta[name="description"]')
            preview = meta.get("content", "") if meta else ""
        return Article(title, url, source, company, publication_date(published).isoformat(),
                       ["LLM", company], re.sub(r"\s+", " ", preview).strip()[:600])
    except (httpx.HTTPError, ValueError, TypeError) as error:
        print(f"  [{source}] error fetching {url}: {error}", flush=True)
        return None


def parse_company_blog(url: str, source: str, *, now: datetime | None = None) -> list[Article]:
    now = now or datetime.now(timezone.utc)
    selector, prefix, company = HTML_BLOGS[url]
    with httpx.Client(follow_redirects=True, timeout=20, headers={"User-Agent": "DigestBot/1.0"}) as client:
        response = client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        candidates = {}
        for anchor in soup.select(selector):
            target = urljoin(url, anchor.get("href", ""))
            parts = urlsplit(target)
            if parts.netloc != urlsplit(url).netloc or not parts.path.startswith(prefix):
                continue
            if not parts.path[len(prefix):].strip("/") or "/research/team/" in parts.path:
                continue
            if target in candidates:
                continue
            card = anchor
            published = _visible_date(card)
            # Meta's text link and date share a card with an image link to the same article.
            if source == "meta":
                for parent in list(anchor.parents)[:3]:
                    published = _visible_date(parent)
                    if published:
                        card = parent
                        break
            if published and not is_recent_publication(published, now):
                continue
            # OpenAI listing dates omit the year. Use month/day only to skip
            # definitely irrelevant cards; verify the full date on the article.
            if source == "openai":
                hint = next((text for text in anchor.stripped_strings if _MONTH_DAY.fullmatch(text)), "")
                if hint:
                    possible = {(now - timedelta(days=i)).strftime("%b %-d") for i in range(8)}
                    if hint not in possible:
                        continue
            paragraph = card.select_one("p")
            preview = paragraph.get_text(" ", strip=True) if paragraph else ""
            candidates[target] = (preview, published)
        if len(candidates) > 40:
            print(f"  [{source}] limiting article page checks to 40", flush=True)
        work = list(candidates.items())[:40]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda item: _fetch_article(
                client, item[0], source, company, item[1][0], now, item[1][1],
            ), work))
        return [article for article in results if article is not None]
