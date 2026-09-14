import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import main
from digest.bot import _build_pages
from digest.config import Config
from digest.models import DigestEntry


class SavedDigestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.digest_path = root / "digest_output.json"
        self.response_path = root / "llm_response.txt"
        for name, path in (("DIGEST_OUTPUT_FILE", self.digest_path), ("LLM_RESPONSE_FILE", self.response_path)):
            patcher = patch.object(main, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.entries = [DigestEntry("Основная статья", "https://example.com/top", "habr", "Автор", [], "Саммари", "LLM")]
        self.mentions = [DigestEntry("Рекомендация", "https://example.com/mention", "habr", "", [], "Дополнение")]

    def test_both_sections_survive_save_load_and_test_send(self):
        main._save_digest(self.entries, self.mentions)
        self.assertEqual(main._load_saved_digest(), (self.entries, self.mentions))
        with patch.object(main, "send_digest", new_callable=AsyncMock) as send:
            main._test_send(Config(tg_bot_token="test-token", tg_chat_id="123", tg_channel_id="456"))
        send.assert_awaited_once_with(self.entries, "test-token", ["123"], self.mentions)
        self.assertIn("Также может быть интересно", "\n".join(_build_pages(self.entries, self.mentions)))

    def test_legacy_array_recovers_matching_mentions_without_llm(self):
        main._save_digest(self.entries, self.mentions)
        full = json.loads(self.digest_path.read_text())
        self.response_path.write_text(json.dumps(full), encoding="utf-8")
        self.digest_path.write_text(json.dumps(full["top"]), encoding="utf-8")
        self.assertEqual(main._load_saved_digest(), (self.entries, self.mentions))
        self.assertIn("honorable_mentions", json.loads(self.digest_path.read_text()))

    def test_legacy_array_does_not_mix_different_digests(self):
        main._save_digest(self.entries, self.mentions)
        full = json.loads(self.digest_path.read_text())
        self.digest_path.write_text(json.dumps(full["top"]), encoding="utf-8")
        full["top"][0]["summary"] = "Другое саммари"
        self.response_path.write_text(json.dumps(full), encoding="utf-8")
        self.assertEqual(main._load_saved_digest(), (self.entries, []))

    def test_legacy_array_works_without_saved_response(self):
        main._save_digest(self.entries, [])
        full = json.loads(self.digest_path.read_text())
        self.digest_path.write_text(json.dumps(full["top"]), encoding="utf-8")
        self.assertEqual(main._load_saved_digest(), (self.entries, []))


if __name__ == "__main__":
    unittest.main()
