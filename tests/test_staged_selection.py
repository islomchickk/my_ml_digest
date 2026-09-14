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
        complete = Mock(side_effect=[answer(items[:5])] + [answer(invalid)] * 3)
        with self.assertRaisesRegex(ValueError, "stage 2/3 failed after 3 attempts.*expected 5 valid articles"):
            select_in_stages(items, complete)
        self.assertEqual(complete.call_count, 4)

    def test_short_second_half_is_completed_without_repeating_valid_choices(self):
        items = articles(20)
        complete = Mock(side_effect=[answer(items[:5]), answer(items[5:9]),
                                     answer(items[9:10]), answer(items[10:14], True)])
        top, mentions = select_in_stages(items, complete)
        self.assertEqual([e.url for e in top], [a.url for a in items[:10]])
        self.assertEqual(len(mentions), 4)
        retry = complete.call_args_list[2]
        self.assertEqual(retry.args[0], 2)
        self.assertEqual(retry.args[3]["properties"]["articles"]["minItems"], 1)
        self.assertEqual({a["url"] for a in json.loads(retry.args[2])},
                         {a.url for a in items[9:]})
        for item in items[:9]:
            self.assertIn(item.url, retry.args[1])
        self.assertIn("expected 5 valid articles, received 4", retry.args[1])

    def test_malformed_json_retry_retains_choices_from_previous_attempt(self):
        items = articles(15)
        complete = Mock(side_effect=[answer(items[:4]), "not JSON", answer(items[4:5]),
                                     answer(items[5:10]), answer([], True)])
        top, mentions = select_in_stages(items, complete)
        self.assertEqual(len(top), 10)
        self.assertEqual(mentions, [])
        for retry in complete.call_args_list[1:3]:
            self.assertEqual(retry.args[3]["properties"]["articles"]["maxItems"], 1)
            self.assertEqual({a["url"] for a in json.loads(retry.args[2])},
                             {a.url for a in items[4:]})

    def test_invalid_recommendations_retry_but_empty_recommendations_stop(self):
        items = articles(15)
        unknown = Article("Unknown", "https://unknown.com/a", "habr", "", "")
        complete = Mock(side_effect=[answer(items[:5]), answer(items[5:10]),
                                     answer([unknown], True), answer([], True)])
        top, mentions = select_in_stages(items, complete)
        self.assertEqual((len(top), len(mentions)), (10, 0))
        self.assertEqual([c.args[0] for c in complete.call_args_list], [1, 2, 3, 3])

    def test_attempt_setting_is_bounded_and_one_attempt_disables_retries(self):
        items = articles(5)
        for count in (0, 11):
            complete = Mock()
            with self.assertRaisesRegex(ValueError, "MAX_ATTEMPTS"):
                select_in_stages(items, complete, max_attempts=count)
            complete.assert_not_called()
        complete = Mock(return_value=answer(items[:4]))
        with self.assertRaisesRegex(ValueError, "after 1 attempts"):
            select_in_stages(items, complete, max_attempts=1)
        self.assertEqual(complete.call_count, 1)

    def test_duplicate_input_urls_do_not_inflate_required_selection(self):
        items = articles(3)
        complete = Mock(return_value=answer(items))
        top, mentions = select_in_stages([items[0], items[1], items[0], items[2]], complete)
        self.assertEqual([e.url for e in top], [a.url for a in items])
        self.assertEqual(mentions, [])
        self.assertEqual(complete.call_count, 1)


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

    def test_retry_snapshots_preserve_both_failed_and_repaired_response(self):
        items = articles(10)
        short, repair = answer(items[5:9]), answer(items[9:10])
        provider = SimpleNamespace(complete=Mock(side_effect=[answer(items[:5]), short, repair]),
                                   last_response_json='{"model": "test"}')
        with patch.object(main, "collect_articles", return_value=items), \
                patch.object(main, "get_provider", return_value=provider):
            top, mentions = main.generate_digest(Config())
        self.assertEqual((len(top), len(mentions)), (10, 0))
        root = Path(self.directory.name)
        self.assertEqual((root / "llm_response_stage_2_attempt_1.txt").read_text(), short)
        self.assertEqual((root / "llm_response_stage_2_attempt_2.txt").read_text(), repair)
        self.assertEqual((root / "llm_response_stage_2.txt").read_text(), repair)

    def test_exhausted_retry_does_not_overwrite_last_digest(self):
        items = articles(10)
        main.DIGEST_OUTPUT_FILE.write_text("previous digest")
        provider = SimpleNamespace(complete=Mock(side_effect=[answer(items[:5])] + ["not JSON"] * 2),
                                   last_response_json='{"model": "test"}')
        with patch.object(main, "collect_articles", return_value=items), \
                patch.object(main, "get_provider", return_value=provider):
            with self.assertRaisesRegex(ValueError, "stage 2/3 failed after 2 attempts"):
                main.generate_digest(Config(llm_selection_max_attempts=2))
        self.assertEqual(main.DIGEST_OUTPUT_FILE.read_text(), "previous digest")
        self.assertEqual(provider.complete.call_count, 3)


if __name__ == "__main__":
    unittest.main()
