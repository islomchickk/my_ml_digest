import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import httpx
import openai

from digest.config import Config
from digest.llm.base import get_provider
from digest.llm.errors import LLMResponseError
from digest.llm.neuraldeep import NeuralDeepProvider
from digest.llm.openai import OpenAIProvider


class ThinkingTests(unittest.TestCase):
    def request(self, provider, *, retry=False, truncated=False):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            if retry and len(requests) == 1:
                return httpx.Response(400, json={"error": {"message":
                    "47924 токенов входа при пределе 46112 на этом тарифе."}})
            return httpx.Response(200, json={
                "id": "test", "created": 0, "model": "test", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "length" if truncated else "stop",
                             "message": {"role": "assistant", "content": '{"articles": []}',
                                         "reasoning_content": "Thinking"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 8000 if truncated else 50,
                          "total_tokens": 8100 if truncated else 150,
                          "completion_tokens_details": {"reasoning_tokens": 40}},
            })

        provider.client.close()
        provider.client = openai.OpenAI(
            api_key="test-key", base_url="https://api.neuraldeep.ru/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        self.addCleanup(provider.client.close)
        output = io.StringIO()
        prompt = json.dumps([{"title": "Article", "url": "https://example.com/a",
                              "preview": "Useful ML engineering details. " * 100}])
        with redirect_stdout(output):
            if truncated:
                with self.assertRaises(LLMResponseError):
                    provider.complete("System", prompt, {"type": "object"})
            else:
                self.assertEqual(provider.complete("System", prompt, {"type": "object"}),
                                 '{"articles": []}')
        return requests, output.getvalue()

    def test_default_disables_thinking_in_actual_http_body(self):
        requests, log = self.request(NeuralDeepProvider("test-key"))
        body = requests[0]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["max_tokens"], 8000)
        self.assertEqual(body["model"], "qwen3.6-unlim")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertNotIn("reasoning_effort", body)
        self.assertIn("thinking=off", log)

    def test_enabled_thinking_budget_survives_input_retry(self):
        requests, log = self.request(NeuralDeepProvider(
            "test-key", enable_thinking=True, thinking_token_budget=1024,
        ), retry=True)
        self.assertEqual(len(requests), 2)
        for body in requests:
            self.assertEqual(body["chat_template_kwargs"],
                             {"enable_thinking": True, "thinking_token_budget": 1024})
            self.assertEqual(body["max_tokens"], 8000)
        self.assertIn("thinking=on (budget 1024 tokens)", log)

    def test_openai_requests_do_not_include_qwen_settings(self):
        requests, _ = self.request(OpenAIProvider("test-key"))
        self.assertNotIn("chat_template_kwargs", requests[0])

    def test_reasoning_is_logged_and_truncated_json_still_rejected(self):
        _, log = self.request(NeuralDeepProvider("test-key"), truncated=True)
        self.assertIn("output_tokens=8000", log)
        self.assertIn("reasoning_tokens=40", log)
        self.assertIn("reasoning_chars=8", log)
        self.assertIn("answer_chars=16", log)

    def test_config_passes_explicit_settings_to_provider(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "neuraldeep", "NEURALDEEP_API_KEY": "test-key",
            "NEURALDEEP_ENABLE_THINKING": "true",
            "NEURALDEEP_THINKING_TOKEN_BUDGET": "512",
            "NEURALDEEP_MAX_OUTPUT_TOKENS": "8000",
        }):
            provider = get_provider(Config.from_env())
        self.addCleanup(provider.client.close)
        self.assertEqual(provider.extra_body["chat_template_kwargs"],
                         {"enable_thinking": True, "thinking_token_budget": 512})

    def test_invalid_thinking_settings_fail_before_api_call(self):
        for budget in (0, -1, 8000, 9000):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "TOKEN_BUDGET"):
                NeuralDeepProvider("test-key", enable_thinking=True, thinking_token_budget=budget)
        with patch.dict("os.environ", {"NEURALDEEP_ENABLE_THINKING": "ture"}):
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                Config.from_env()


if __name__ == "__main__":
    unittest.main()
