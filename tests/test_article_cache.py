import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import main
from digest.article_cache import ArticleCache, CACHE_TTL_SECONDS
from digest.config import Config
from digest.models import Article, ArticleStats, DigestEntry


def article(index=1):
    return Article(f"Article {index}", f"https://example.com/{index}", "habr", "Author", "",
                   ["ML"], "Preview", ArticleStats(score=10))


class CacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = ArticleCache(self.root / "data" / "articles_cache.json")

    def test_cache_survives_restart_and_expires_at_exactly_24_hours(self):
        with patch("digest.article_cache.time.time", return_value=1000):
            self.cache.save([article()])
        reopened = ArticleCache(self.cache.path)
        for age, fresh in ((0, True), (CACHE_TTL_SECONDS - 1, True),
                           (CACHE_TTL_SECONDS, False), (CACHE_TTL_SECONDS + 1, False), (-1, False)):
            with self.subTest(age=age), patch("digest.article_cache.time.time", return_value=1000 + age):
                loaded = reopened.load_fresh()
            self.assertEqual(loaded is not None, fresh)
            if loaded:
                self.assertEqual(loaded.articles, [article()])
                self.assertEqual(loaded.parsed_at, 1000)
        self.assertEqual(json.loads(self.cache.path.read_text())["parsed_at"], 1000)

    def test_missing_corrupt_and_invalid_cache_are_rejected(self):
        self.assertIsNone(self.cache.load_fresh())
        self.cache.path.parent.mkdir()
        for data in ("not json", "null", "[]", '{"version": 1}',
                     json.dumps({"version": 1, "parsed_at": True, "articles": []}),
                     json.dumps({"version": 1, "parsed_at": float("nan"), "articles": []}),
                     json.dumps({"version": 1, "parsed_at": 1000, "articles": [{}]})):
            with self.subTest(data=data):
                self.cache.path.write_text(data)
                self.assertIsNone(self.cache.load_fresh())

    def pipeline(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, filename in (("ARTICLES_FILE", "articles.json"), ("DIGEST_OUTPUT_FILE", "digest_output.json"),
                               ("LLM_RESPONSE_FILE", "llm_response.txt"), ("LLM_API_RESPONSE_FILE", "llm_response.json")):
            stack.enter_context(patch.object(main, name, self.root / filename))
        stack.enter_context(patch.dict("os.environ", {"DIGEST_DATA_DIR": str(self.cache.path.parent)}))
        stack.enter_context(patch.object(main, "get_provider", return_value=Mock()))
        selected = DigestEntry("Selected", article().url, "habr", "Author", [], "Summary")
        return stack.enter_context(patch.object(main, "select_in_stages", return_value=([selected], [])))

    def test_button_reuses_pool_and_filters_history_without_refreshing_timestamp(self):
        select = self.pipeline()
        old, new = article(0), article(1)
        with patch("digest.article_cache.time.time", return_value=1000):
            self.cache.save([old, new])
        with patch("digest.article_cache.time.time", return_value=1100), \
                patch.object(main, "collect_articles") as collect:
            main.generate_digest(Config(), exclude_urls={old.url}, use_article_cache=True)
            self.assertEqual(select.call_args.args[0], [new])
            # Undoing delivery can expose the old article in this same cached pool.
            main.generate_digest(Config(), exclude_urls=set(), use_article_cache=True)
            self.assertEqual(select.call_args.args[0], [old, new])
        collect.assert_not_called()
        self.assertEqual(json.loads(self.cache.path.read_text())["parsed_at"], 1000)
        self.assertEqual(len(json.loads(main.ARTICLES_FILE.read_text())), 2)

    def test_expired_pool_is_reparsed_and_replaces_persistent_cache(self):
        self.pipeline()
        with patch("digest.article_cache.time.time", return_value=1000):
            self.cache.save([article(0)])
        now = 1000 + CACHE_TTL_SECONDS
        with patch("digest.article_cache.time.time", return_value=now), \
                patch.object(main, "collect_articles", return_value=[article(1)]) as collect:
            main.generate_digest(Config(), use_article_cache=True)
        collect.assert_called_once()
        data = json.loads(self.cache.path.read_text())
        self.assertEqual(data["parsed_at"], now)
        self.assertEqual(data["articles"], [asdict(article(1))])

    def test_regular_run_always_parses_and_seeds_button_cache(self):
        self.pipeline()
        with patch("digest.article_cache.time.time", return_value=1000):
            self.cache.save([article(0)])
        with patch("digest.article_cache.time.time", return_value=1100), \
                patch.object(main, "collect_articles", return_value=[article(1)]) as collect:
            main.generate_digest(Config())
            main.generate_digest(Config(), use_article_cache=True)
        collect.assert_called_once()
        self.assertEqual(json.loads(self.cache.path.read_text())["parsed_at"], 1100)

    def test_failed_parse_preserves_old_cache(self):
        self.pipeline()
        with patch("digest.article_cache.time.time", return_value=1000):
            self.cache.save([article(0)])
        original = self.cache.path.read_text()
        with patch("digest.article_cache.time.time", return_value=1000 + CACHE_TTL_SECONDS), \
                patch.object(main, "collect_articles", side_effect=RuntimeError("RSS failed")):
            with self.assertRaisesRegex(RuntimeError, "RSS failed"):
                main.generate_digest(Config(), use_article_cache=True)
        self.assertEqual(self.cache.path.read_text(), original)

    def test_empty_fresh_cache_skips_both_parsing_and_llm(self):
        self.pipeline()
        with patch("digest.article_cache.time.time", return_value=1000):
            self.cache.save([])
            with patch.object(main, "collect_articles") as collect, patch.object(main, "get_provider") as provider:
                self.assertEqual(main.generate_digest(Config(), use_article_cache=True), ([], []))
        collect.assert_not_called()
        provider.assert_not_called()

    def test_missing_cache_saves_full_pool_even_when_llm_fails(self):
        select = self.pipeline()
        old, new = article(0), article(1)
        select.side_effect = [ValueError("invalid selection"), ([], [])]
        with patch("digest.article_cache.time.time", return_value=1000), \
                patch.object(main, "collect_articles", return_value=[old, new]) as collect:
            with self.assertRaisesRegex(ValueError, "invalid selection"):
                main.generate_digest(Config(), exclude_urls={old.url}, use_article_cache=True)
            main.generate_digest(Config(), exclude_urls={old.url}, use_article_cache=True)
        collect.assert_called_once()
        self.assertEqual(select.call_args_list[0].args[0], [new])
        self.assertEqual(select.call_args_list[1].args[0], [new])
        self.assertEqual(len(json.loads(self.cache.path.read_text())["articles"]), 2)

    def test_bot_callback_enables_cache(self):
        async def bot(config, generate, store):
            generate({article(0).url})

        with patch("sys.argv", ["main.py", "--bot"]), \
                patch.object(main.Config, "from_env", return_value=Config()), \
                patch.object(main, "DigestStore", return_value=Mock()), \
                patch.object(main, "run_bot", side_effect=bot), \
                patch.object(main, "generate_digest", return_value=([], [])) as generate:
            main.main()
        self.assertTrue(generate.call_args.kwargs["use_article_cache"])
        self.assertEqual(generate.call_args.kwargs["exclude_urls"], {article(0).url})


if __name__ == "__main__":
    unittest.main()
