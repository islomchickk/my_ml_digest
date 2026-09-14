import json
import unittest
from unittest.mock import patch

import httpx
import openai

from digest.llm.neuraldeep import NeuralDeepProvider
from digest.llm.openai import OpenAIProvider
from digest.llm.token_budget import count_tokens, fit_article_prompt, request_overhead


def articles(count=20, preview="Практическое руководство по обучению моделей. " * 40):
    return [
        {"title": f"Статья {i}", "url": f"https://example.com/{i}",
         "source": "habr", "author": "Автор", "tags": ["ML"],
         "preview": preview, "stats": {"score": i}}
        for i in range(count)
    ]


def bad_request(message):
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.com")),
        body={"error": {"message": message}},
    )


class TokenBudgetTests(unittest.TestCase):
    def test_short_request_preserves_all_fields(self):
        items = articles(2, 'Кавычки " и эмодзи 🦊 <|endoftext|>')
        result = fit_article_prompt(json.dumps(items), 100, 1000)
        self.assertEqual(json.loads(result.text), items)
        self.assertEqual(result.estimated_tokens, 100 + count_tokens(result.text))

    def test_large_request_shortens_previews_without_losing_metadata(self):
        items = articles()
        result = fit_article_prompt(json.dumps(items), 100, 3000)
        trimmed = json.loads(result.text)
        self.assertEqual(len(trimmed), len(items))
        self.assertLessEqual(result.estimated_tokens, 3000)
        for original, item in zip(items, trimmed):
            self.assertTrue(original["preview"].startswith(item["preview"]))
            self.assertLess(len(item["preview"]), len(original["preview"]))
            self.assertEqual(
                {k: v for k, v in original.items() if k != "preview"},
                {k: v for k, v in item.items() if k != "preview"},
            )

    def test_metadata_overflow_keeps_complete_records_in_order(self):
        items = articles(100, "")
        result = fit_article_prompt(json.dumps(items), 100, 1000)
        self.assertGreater(result.article_count, 0)
        self.assertLess(result.article_count, len(items))
        self.assertEqual(json.loads(result.text), items[:result.article_count])
        self.assertLessEqual(result.estimated_tokens, 1000)

    def test_impossible_budget_fails_before_request(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            fit_article_prompt(json.dumps(articles(1)), 100, 100)

    def test_schema_and_system_are_included_in_budget(self):
        schema = {"type": "object", "description": "Русская инструкция " * 20}
        overhead = request_overhead("System instruction", schema)
        self.assertGreater(overhead, request_overhead("System instruction", None))
        result = fit_article_prompt(json.dumps(articles()), overhead, 3000)
        self.assertLessEqual(count_tokens(result.text) + overhead, 3000)


class NeuralDeepRetryTests(unittest.TestCase):
    def setUp(self):
        self.provider = NeuralDeepProvider("test-key")
        self.prompt = json.dumps(articles(30))
        self.schema = {"type": "object"}

    def tearDown(self):
        self.provider.client.close()

    def test_server_count_reduces_request_and_preserves_instructions(self):
        error = bad_request(
            "Слишком длинный запрос: 47924 токенов входа при пределе 46112 на этом тарифе."
        )
        with patch.object(OpenAIProvider, "complete", side_effect=[error, "{}"]) as complete:
            result = self.provider.complete("System", self.prompt, self.schema)
        self.assertEqual(result, "{}")
        first, second = complete.call_args_list
        self.assertLess(count_tokens(second.args[1]), count_tokens(first.args[1]))
        self.assertEqual(second.args[0], "System")
        self.assertEqual(second.args[2], self.schema)
        self.assertEqual(len(json.loads(second.args[1])), 30)

    def test_other_bad_requests_are_not_retried(self):
        error = bad_request("Unsupported response_format")
        with patch.object(OpenAIProvider, "complete", side_effect=error) as complete:
            with self.assertRaises(openai.BadRequestError):
                self.provider.complete("System", self.prompt)
        self.assertEqual(complete.call_count, 1)

    def test_input_limit_retries_are_bounded(self):
        error = bad_request("47924 токенов входа при пределе 46112 на этом тарифе.")
        with patch.object(OpenAIProvider, "complete", side_effect=error) as complete:
            with self.assertRaises(openai.BadRequestError):
                self.provider.complete("System", self.prompt)
        self.assertEqual(complete.call_count, 3)


if __name__ == "__main__":
    unittest.main()
