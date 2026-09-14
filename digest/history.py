"""Persistent delivery history, pagination and a cross-process generation lock."""

import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
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
            connection.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?)",
                               (str(chat_id), message_id, json.dumps(pages, ensure_ascii=False)))
            connection.executemany("INSERT OR IGNORE INTO sent VALUES (?, ?)",
                                   [(str(chat_id), article_key(entry.url)) for entry in entries])

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
