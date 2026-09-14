"""Persistent delivery history, pagination and a cross-process generation lock."""

import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from digest.models import DigestEntry


def article_key(url: str) -> str:
    parts = urlsplit(url.strip())
    medium_host = parts.hostname == "medium.com" or (parts.hostname or "").endswith(".medium.com")
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
             and not (medium_host and key.lower() == "source")]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                      parts.path.rstrip("/"), urlencode(sorted(query)), ""))


class DigestBusyError(RuntimeError):
    pass


class _ArticleLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls: set[str] = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.urls.add(article_key(value))


class DigestStore:
    def __init__(self, path: Path | None = None):
        self.path = path or Path(os.getenv("DIGEST_DATA_DIR", "data")) / "digest.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sent (
                    chat_id TEXT NOT NULL, url TEXT NOT NULL,
                    PRIMARY KEY (chat_id, url)
                );
                CREATE TABLE IF NOT EXISTS pages (
                    chat_id TEXT NOT NULL, message_id INTEGER NOT NULL,
                    content TEXT NOT NULL, PRIMARY KEY (chat_id, message_id)
                );
            """)
            connection.execute("BEGIN IMMEDIATE")
            migrate = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='delivery_batches'"
            ).fetchone() is None
            connection.execute("""
                CREATE TABLE IF NOT EXISTS delivery_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL, message_id INTEGER NOT NULL,
                    urls TEXT NOT NULL, new_urls TEXT NOT NULL,
                    forgotten INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (chat_id, message_id)
                )
            """)
            if migrate:
                self._migrate_deliveries(connection)

    @staticmethod
    def _migrate_deliveries(connection) -> None:
        # Old versions saved every delivered page, including the recommendations.
        # Reconstruct batches from those pages, never from a possibly unsent file.
        deliveries = []
        all_urls: dict[str, set[str]] = {}
        for chat, message, content in connection.execute(
            "SELECT chat_id, message_id, content FROM pages ORDER BY chat_id, message_id"
        ).fetchall():
            links = _ArticleLinks()
            for page in json.loads(content):
                links.feed(page)
            deliveries.append((chat, message, links.urls))
            all_urls.setdefault(chat, set()).update(links.urls)
        known = {
            chat: {row[0] for row in connection.execute(
                "SELECT url FROM sent WHERE chat_id = ?", (chat,),
            )}
            for chat in all_urls
        }
        previous = {chat: urls - all_urls[chat] for chat, urls in known.items()}
        for chat, message, urls in deliveries:
            new_urls = (urls & known[chat]) - previous[chat]
            connection.execute(
                "INSERT INTO delivery_batches (chat_id, message_id, urls, new_urls) VALUES (?, ?, ?, ?)",
                (chat, message, json.dumps(sorted(urls)), json.dumps(sorted(new_urls))),
            )
            previous[chat].update(urls)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def sent_urls(self, chat_id: str) -> set[str]:
        with self.connect() as connection:
            return {row[0] for row in connection.execute(
                "SELECT url FROM sent WHERE chat_id = ?", (str(chat_id),),
            )}

    def remember(self, chat_id: str, entries: list[DigestEntry]) -> None:
        with self.connect() as connection:
            connection.executemany("INSERT OR IGNORE INTO sent VALUES (?, ?)",
                                   [(str(chat_id), article_key(entry.url)) for entry in entries])

    def record_delivery(self, chat_id: str, message_id: int, pages: list[str], entries: list[DigestEntry]) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chat_id = str(chat_id)
            urls = {article_key(entry.url) for entry in entries}
            existing_batch = connection.execute(
                "SELECT 1 FROM delivery_batches WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            ).fetchone()
            if existing_batch:
                return
            previous = {row[0] for row in connection.execute(
                "SELECT url FROM sent WHERE chat_id = ?", (chat_id,),
            )}
            connection.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?)",
                               (chat_id, message_id, json.dumps(pages, ensure_ascii=False)))
            connection.executemany("INSERT OR IGNORE INTO sent VALUES (?, ?)",
                                   [(chat_id, url) for url in urls])
            connection.execute(
                "INSERT INTO delivery_batches (chat_id, message_id, urls, new_urls) VALUES (?, ?, ?, ?)",
                (chat_id, message_id, json.dumps(sorted(urls)), json.dumps(sorted(urls - previous))),
            )

    def forget_last_delivery(self, chat_id: str) -> tuple[int, int] | None:
        """Undo the latest active batch; preserve earlier history and pagination."""
        with self.generation_lock(), self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id, urls, new_urls FROM delivery_batches "
                "WHERE chat_id = ? AND forgotten = 0 ORDER BY id DESC LIMIT 1",
                (str(chat_id),),
            ).fetchone()
            if row is None:
                return None
            batch_id, urls, new_urls = row
            connection.executemany(
                "DELETE FROM sent WHERE chat_id = ? AND url = ?",
                [(str(chat_id), url) for url in json.loads(new_urls)],
            )
            removed = connection.total_changes
            connection.execute("UPDATE delivery_batches SET forgotten = 1 WHERE id = ?", (batch_id,))
            return len(json.loads(urls)), removed

    def load_pages(self, chat_id: str, message_id: int) -> list[str] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT content FROM pages WHERE chat_id = ? AND message_id = ?",
                                     (str(chat_id), message_id)).fetchone()
            return json.loads(row[0]) if row else None

    @contextmanager
    def generation_lock(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise DigestBusyError("A digest is already being generated") from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
