import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from digest.bot import _build_pages
from digest.config import Config
from digest.history import DigestBusyError, DigestStore, article_key
from digest.models import DigestEntry


def entry(index):
    return DigestEntry(f"Article {index}", f"https://example.com/{index}",
                       "habr", "Author", [], "Summary", "LLM")


class ForgetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "digest.sqlite3"
        self.store = DigestStore(self.path)

    def test_forgets_both_sections_preserves_earlier_chat_and_pages(self):
        self.store.record_delivery("123", 1, ["old"], [entry(0)])
        top, mentions = [entry(i) for i in range(1, 11)], [entry(i) for i in range(11, 15)]
        pages = _build_pages(top, mentions)
        self.store.record_delivery("123", 2, pages, top + mentions)
        self.store.record_delivery("456", 3, pages, top + mentions)
        self.assertEqual(self.store.forget_last_delivery("123"), (14, 14))
        self.assertEqual(self.store.sent_urls("123"), {entry(0).url})
        self.assertEqual(len(self.store.sent_urls("456")), 14)
        self.assertEqual(self.store.load_pages("123", 2), pages)
        restarted = DigestStore(self.path)
        self.assertEqual(restarted.forget_last_delivery("123"), (1, 1))
        self.assertIsNone(restarted.forget_last_delivery("123"))

    def test_repeated_articles_and_imported_history_are_preserved(self):
        self.store.remember("123", [entry(0)])
        self.store.record_delivery("123", 1, ["first"], [entry(1)])
        self.store.record_delivery("123", 2, ["second"], [entry(0), entry(1), entry(2)])
        self.assertEqual(self.store.forget_last_delivery("123"), (3, 1))
        self.assertEqual(self.store.sent_urls("123"), {entry(0).url, entry(1).url})

    def test_duplicate_delivery_record_does_not_create_another_batch(self):
        self.store.record_delivery("123", 1, ["page"], [entry(1)])
        self.store.record_delivery("123", 1, ["page"], [entry(1)])
        self.assertEqual(self.store.forget_last_delivery("123"), (1, 1))
        self.assertIsNone(self.store.forget_last_delivery("123"))

    def test_busy_generation_blocks_reset_without_changes(self):
        self.store.record_delivery("123", 1, ["page"], [entry(1)])
        other = DigestStore(self.path)
        with self.store.generation_lock(), self.assertRaises(DigestBusyError):
            other.forget_last_delivery("123")
        self.assertEqual(self.store.sent_urls("123"), {entry(1).url})
        self.assertEqual(other.forget_last_delivery("123"), (1, 1))

    def test_old_database_migrates_delivered_pages_once(self):
        legacy = self.path.with_name("legacy.sqlite3")
        tracked = entry(2)
        tracked.url += "?a=1&b=2&utm_source=tg#part"
        with sqlite3.connect(legacy) as db:
            db.executescript("""
                CREATE TABLE sent (chat_id TEXT, url TEXT, PRIMARY KEY(chat_id, url));
                CREATE TABLE pages (chat_id TEXT, message_id INTEGER, content TEXT,
                                    PRIMARY KEY(chat_id, message_id));
            """)
            db.executemany("INSERT INTO sent VALUES (?, ?)",
                           [("123", article_key(e.url)) for e in [entry(0), entry(1), tracked]])
            # Insert out of chronological order; URLs contain escaped ampersands.
            db.executemany("INSERT INTO pages VALUES (?, ?, ?)", [
                ("123", 20, json.dumps(_build_pages([entry(1)], [tracked]))),
                ("123", 10, json.dumps(_build_pages([entry(1)]))),
            ])
        store = DigestStore(legacy)
        self.assertEqual(store.forget_last_delivery("123"), (2, 1))
        self.assertEqual(store.sent_urls("123"), {entry(0).url, entry(1).url})
        restarted = DigestStore(legacy)
        self.assertEqual(restarted.forget_last_delivery("123"), (1, 1))
        self.assertEqual(restarted.sent_urls("123"), {entry(0).url})

    def test_cli_resets_delivery_without_reading_saved_file_or_sending(self):
        self.store.record_delivery("123", 1, ["page"], [entry(1)])
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--forget-last-sent"]), \
                patch.object(main.Config, "from_env", return_value=Config(tg_chat_id="123")), \
                patch.object(main, "DigestStore", return_value=self.store), \
                patch.object(main, "_load_saved_digest") as load, \
                patch.object(main, "generate_digest") as generate, \
                patch.object(main, "send_digest") as send, redirect_stdout(output):
            main.main()
        load.assert_not_called()
        generate.assert_not_called()
        send.assert_not_called()
        self.assertIn("removed 1 new URLs", output.getvalue())
        self.assertEqual(self.store.sent_urls("123"), set())

    def test_cli_requires_personal_chat_and_rejects_conflicting_modes(self):
        with patch("sys.argv", ["main.py", "--forget-last-sent"]), \
                patch.object(main.Config, "from_env", return_value=Config(tg_channel_id="-456")), \
                patch.object(main, "DigestStore", return_value=self.store), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main.main()
        self.assertEqual(error.exception.code, 1)
        with patch("sys.argv", ["main.py", "--forget-last-sent", "--test-send"]), \
                patch("sys.stderr", io.StringIO()), patch.object(main, "send_digest") as send:
            with self.assertRaises(SystemExit) as error:
                main.main()
        self.assertEqual(error.exception.code, 2)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
