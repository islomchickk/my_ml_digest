import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from digest.config import Config
from digest.llm.errors import LLMResponseError
from digest.llm.selection import select_in_stages
from digest.models import Article, ArticleStats


def articles(count):
    return [Article(f"Article {i}", f"https://example.com/{i}", "habr", f"Author {i}", "",
                    ["ML"], f"Preview {i}", ArticleStats(score=i)) for i in range(count)]


def answer(items, mentions=False):
    return json.dumps({"articles": [
        {"title": item.title, "url": item.url, "source": item.source,
         "summary": f"Summary for {item.title}",
         **({} if mentions else {"author": item.author, "tags": item.tags, "category": "LLM (RAG)"})}
        for item in items
    ]})


class StagedSelectionTests(unittest.TestCase):
    def test_three_requests_share_selection_and_remove_candidates(self):
        items = articles(20)
        complete = Mock(side_effect=[answer(items[:5]), answer(items[5:10]), answer(items[10:14], True)])
        top, mentions = select_in_stages(items, complete)
        self.assertEqual([entry.url for entry in top], [item.url for item in items[:10]])
        self.assertEqual([entry.url for entry in mentions], [item.url for item in items[10:14]])
        for stage, call in enumerate(complete.call_args_list, 1):
            self.assertEqual(call.args[0], stage)
            candidate_urls = {item["url"] for item in json.loads(call.args[2])}
            previous = items[:5 * (stage - 1)]
            self.assertEqual(candidate_urls, {item.url for item in items[5 * (stage - 1):]})
            for item in previous:
                self.assertIn(item.url, call.args[1])
                self.assertIn(f"Summary for {item.title}", call.args[1])
                self.assertNotIn(item.url, candidate_urls)
            self.assertEqual(call.args[3]["properties"]["articles"]["maxItems"], 4 if stage == 3 else 5)
        self.assertEqual(top[2].stats.score, 2)

    def test_small_pools_skip_requests_and_do_not_invent_articles(self):
        for size, expected_calls in ((0, 0), (3, 1), (7, 2), (12, 3)):
            with self.subTest(size=size):
                items = articles(size)
                replies = [answer(items[:5]), answer(items[5:10]), answer(items[10:14], True)]
                complete = Mock(side_effect=replies)
                top, mentions = select_in_stages(items, complete)
                self.assertEqual(len(top), min(size, 10))
                self.assertEqual(len(mentions), max(0, size - 10))
                self.assertEqual(complete.call_count, expected_calls)

    def test_empty_optional_recommendations_are_allowed(self):
        items = articles(15)
        complete = Mock(side_effect=[answer(items[:5]), answer(items[5:10]), answer([], True)])
        top, mentions = select_in_stages(items, complete)
        self.assertEqual(len(top), 10)
        self.assertEqual(mentions, [])

    def test_repeated_unknown_and_duplicate_choices_cannot_fill_second_half(self):
        items = articles(15)
        invalid = [items[0], items[5], items[5], Article("Unknown", "https://unknown.com/a", "habr", "", "")]
        complete = Mock(side_effect=[answer(items[:5]), answer(invalid)])
        with self.assertRaisesRegex(ValueError, "stage 2/3.*expected 5 valid articles"):
            select_in_stages(items, complete)
        self.assertEqual(complete.call_count, 2)


class StageDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        for name, filename in (("ARTICLES_FILE", "articles.json"), ("DIGEST_OUTPUT_FILE", "digest_output.json"),
                               ("LLM_RESPONSE_FILE", "llm_response.txt"), ("LLM_API_RESPONSE_FILE", "llm_response.json")):
            patcher = patch.object(main, name, root / filename)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_success_saves_three_responses_and_combines_both_sections(self):
        items = articles(20)
        provider = SimpleNamespace(complete=Mock(side_effect=[answer(items[:5]), answer(items[5:10]), answer(items[10:14], True)]),
                                   last_response_json='{"model": "test"}')
        with patch.object(main, "collect_articles", return_value=items), patch.object(main, "get_provider", return_value=provider):
            top, mentions = main.generate_digest(Config())
        self.assertEqual((len(top), len(mentions)), (10, 4))
        self.assertEqual(tuple(map(len, main._load_saved_digest())), (10, 4))
        combined = json.loads(main.LLM_RESPONSE_FILE.read_text())
        self.assertEqual((len(combined["top"]), len(combined["honorable_mentions"])), (10, 4))
        for stage in range(1, 4):
            self.assertTrue((Path(self.directory.name) / f"llm_response_stage_{stage}.txt").exists())
            self.assertTrue((Path(self.directory.name) / f"llm_response_stage_{stage}.json").exists())

    def test_second_stage_truncation_preserves_previous_digest_and_diagnostics(self):
        items = articles(20)
        main.DIGEST_OUTPUT_FILE.write_text("previous digest")
        error = LLMResponseError("truncated", '{"finish_reason": "length"}')
        provider = SimpleNamespace(complete=Mock(side_effect=[answer(items[:5]), error]),
                                   last_response_json='{"finish_reason": "stop"}')
        with patch.object(main, "collect_articles", return_value=items), patch.object(main, "get_provider", return_value=provider):
            with self.assertRaises(LLMResponseError):
                main.generate_digest(Config())
        self.assertEqual(main.DIGEST_OUTPUT_FILE.read_text(), "previous digest")
        self.assertEqual(provider.complete.call_count, 2)
        root = Path(self.directory.name)
        self.assertEqual(json.loads((root / "llm_response_stage_2.json").read_text()), {"finish_reason": "length"})
        self.assertTrue((root / "llm_response_stage_1.txt").exists())
        self.assertFalse((root / "llm_response_stage_3.txt").exists())


if __name__ == "__main__":
    unittest.main()
