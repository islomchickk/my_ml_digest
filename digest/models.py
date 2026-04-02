from dataclasses import dataclass, field


@dataclass
class ArticleStats:
    score: int = 0
    votes_plus: int = 0
    votes_minus: int = 0
    comments: int = 0
    bookmarks: int = 0
    views: int = 0
    reach: int = 0


@dataclass
class Article:
    title: str
    url: str
    source: str
    author: str
    published: str
    tags: list[str] = field(default_factory=list)
    preview: str = ""
    stats: ArticleStats = field(default_factory=ArticleStats)


@dataclass
class DigestEntry:
    """Статья после обработки LLM — с саммари."""
    title: str
    url: str
    source: str
    author: str
    tags: list[str]
    summary: str
    category: str = ""
    stats: ArticleStats | None = None
