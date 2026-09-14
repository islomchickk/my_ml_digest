import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion

import main
from digest.config import Config
from digest.llm.errors import LLMResponseError
from digest.llm.openai import OpenAIProvider
from digest.llm.prompt import parse_llm_response


ENTRY = {"title": "Статья", "url": "https://example.com/article", "source": "habr",
         "author": "Автор", "tags": ["ML"], "summary": "Полезное руководство.", "category": "ML"}
ANSWER = json.dumps({"top": [ENTRY], "honorable_mentions": []}, ensure_ascii=False)


def completion(content=None, finish_reason="stop", refusal=None, choices=True):
    return ChatCompletion.model_validate({
        "id": "test", "created": 0, "model": "test", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content,
                                 "refusal": refusal, "reasoning_content": "Thinking..."}}] if choices else [],
    })


class ParseResponseTests(unittest.TestCase):
    def test_recognized_wrappers_preserve_digest(self):
        for response in (ANSWER, f"```json\n{ANSWER}\n```", f"```\n{ANSWER}\n```",
                         f"<think>Consider {{this}} first.</think>\n```json\n{ANSWER}\n```",
                         "\ufeff" + ANSWER):
            with self.subTest(response=response[:30]):
                entries, mentions = parse_llm_response(response, [])
                self.assertEqual(entries[0].url, ENTRY["url"])
                self.assertEqual(entries[0].summary, ENTRY["summary"])
                self.assertEqual(mentions, [])

    def test_legacy_array_and_object_still_work(self):
        for data in ([ENTRY], {"articles": [ENTRY]}):
            self.assertEqual(len(parse_llm_response(json.dumps(data), [])[0]), 1)

    def test_empty_malformed_and_unexpected_responses_are_rejected(self):
        for response in ("", " ", "<think>unfinished", "<think>finished</think>",
                         ANSWER[:-5], "Here is the answer: " + ANSWER, "{}", "null", "[]",
                         '{"top": [], "honorable_mentions": []}',
                         '{"top": [{}]}', '{"top": "wrong"}',
                         json.dumps({"top": [{**ENTRY, "tags": "wrong"}]}),
                         json.dumps({"top": [ENTRY], "honorable_mentions": None})):
            with self.subTest(response=response[:30]), self.assertRaises(ValueError):
                parse_llm_response(response, [])


class ProviderResponseTests(unittest.TestCase):
    def setUp(self):
        self.provider = OpenAIProvider("test-key")

    def tearDown(self):
        self.provider.client.close()

    def test_valid_answer_is_returned_and_api_metadata_retained(self):
        with patch.object(self.provider.client.chat.completions, "create", return_value=completion(ANSWER)):
            self.assertEqual(self.provider.complete("System", "User"), ANSWER)
        self.assertIn("reasoning_content", self.provider.last_response_json)

    def test_unusable_completions_raise_with_original_response(self):
        cases = [(completion(), "empty"), (completion(" "), "empty"),
                 (completion(ANSWER[:-5], "length"), "truncated"),
                 (completion(refusal="Cannot answer"), "refused"),
                 (completion(choices=False), "no choices"),
                 (completion("Filtered", "content_filter"), "incomplete")]
        for response, message in cases:
            with self.subTest(message=message):
                with patch.object(self.provider.client.chat.completions, "create", return_value=response):
                    with self.assertRaisesRegex(LLMResponseError, message) as caught:
                        self.provider.complete("System", "User")
                self.assertEqual(json.loads(caught.exception.raw_response), response.model_dump())


class PipelineDiagnosticsTests(unittest.TestCase):
    def test_failed_answers_are_saved_and_not_sent(self):
        for error in (False, True):
            with self.subTest(provider_error=error), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                article_file = root / "articles.json"
                article_file.write_text(json.dumps([{**ENTRY, "published": "2026-09-14"}]), encoding="utf-8")
                stack.enter_context(patch.object(main, "ARTICLES_FILE", article_file))
                stack.enter_context(patch.object(main, "LLM_RESPONSE_FILE", root / "llm_response.txt"))
                stack.enter_context(patch.object(main, "LLM_API_RESPONSE_FILE", root / "llm_response.json"))
                stack.enter_context(patch("sys.argv", ["main.py", "--no-parse"]))
                stack.enter_context(patch.object(main.Config, "from_env", return_value=Config()))
                provider = Mock()
                provider.last_response_json = '{"test": true}'
                if error:
                    provider.complete.side_effect = LLMResponseError("empty content", '{"test": true}')
                else:
                    provider.complete.return_value = "Not JSON"
                stack.enter_context(patch.object(main, "get_provider", return_value=provider))
                send = stack.enter_context(patch.object(main, "send_digest"))
                with self.assertRaises(SystemExit) as caught:
                    main.main()
                self.assertEqual(caught.exception.code, 1)
                self.assertEqual(json.loads((root / "llm_response.json").read_text()), {"test": True})
                if not error:
                    self.assertEqual((root / "llm_response.txt").read_text(), "Not JSON")
                send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
