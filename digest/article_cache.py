"""Persist the complete parsed article pool independently of delivery history."""

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from digest.models import Article, ArticleStats

CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_VERSION = 2  # Invalidate pools collected before official company sources existed.


@dataclass
class CachedArticles:
    articles: list[Article]
    parsed_at: float


class ArticleCache:
    def __init__(self, path: Path | None = None):
        self.path = path or Path(os.getenv("DIGEST_DATA_DIR", "data")) / "articles_cache.json"

    def load_fresh(self) -> CachedArticles | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["version"] != CACHE_VERSION:
                return None
            parsed_at = data["parsed_at"]
            if (not isinstance(parsed_at, (int, float)) or isinstance(parsed_at, bool)
                    or not math.isfinite(parsed_at)
                    or not 0 <= time.time() - parsed_at < CACHE_TTL_SECONDS):
                return None
            if not isinstance(data["articles"], list):
                return None
            articles = []
            for item in data["articles"]:
                if any(not isinstance(item[key], str) for key in
                       ("title", "url", "source", "author", "published", "preview")):
                    return None
                if (not isinstance(item["tags"], list)
                        or any(not isinstance(tag, str) for tag in item["tags"])):
                    return None
                stats = ArticleStats(**item["stats"])
                if any(not isinstance(value, int) or isinstance(value, bool)
                       for value in asdict(stats).values()):
                    return None
                articles.append(Article(**{**item, "stats": stats}))
            return CachedArticles(articles, parsed_at)
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def save(self, articles: list[Article]) -> None:
        data = {"version": CACHE_VERSION, "parsed_at": time.time(), "articles": [asdict(a) for a in articles]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix="articles_cache_", suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                json.dump(data, file, ensure_ascii=False)
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
