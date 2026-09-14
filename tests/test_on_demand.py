import asyncio
import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import main
from digest.bot import _build_keyboard, _send_to_chat, handle_digest_now, handle_page, send_digest
from digest.bot import run_bot
from digest.config import Config
from digest.history import DigestStore, DigestBusyError, article_key
from digest.models import Article, DigestEntry
from digest.parser import collect_articles
from aiogram import Bot, Dispatcher
from aiogram.types import Update, Message
from aiogram.methods import SendMessage, AnswerCallbackQuery


def entry(url="https://example.com/new"):
    return DigestEntry("Статья", url, "habr", "Автор", [], "Саммари", "LLM")


def callback(chat_id=123, user_id=123, message_id=1, data="digest_now"):
    return SimpleNamespace(
        message=SimpleNamespace(chat=SimpleNamespace(id=chat_id), message_id=message_id,
                                edit_text=AsyncMock()),
        from_user=SimpleNamespace(id=user_id), data=data, answer=AsyncMock(),
    )


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "digest.sqlite3"
        self.store = DigestStore(self.path)

    def test_history_and_pages_survive_restart_and_are_per_chat(self):
        self.store.record_delivery("123", 1, ["first", "second"], [entry()])
        self.store.record_delivery("456", 1, ["other chat"], [entry("https://example.com/other")])
        restarted = DigestStore(self.path)
        self.assertEqual(restarted.sent_urls("123"), {"https://example.com/new"})
        self.assertEqual(restarted.load_pages("123", 1), ["first", "second"])
        self.assertEqual(restarted.load_pages("456", 1), ["other chat"])

    def test_tracking_parameters_and_fragments_do_not_allow_repeats(self):
        self.store.remember("123", [entry("https://EXAMPLE.com/new/?utm_source=tg#section")])
        self.assertIn(article_key("https://example.com/new?utm_medium=rss"), self.store.sent_urls("123"))
        self.assertNotEqual(article_key("https://example.com/new?a=1"), article_key("https://example.com/new?a=2"))
        self.assertEqual(article_key("https://medium.com/a?source=rss-tag-ml"), article_key("https://medium.com/a?source=rss-tag-ai"))

    def test_generation_lock_is_shared_and_released_after_failure(self):
        other = DigestStore(self.path)
        with self.assertRaises(ValueError):
            with self.store.generation_lock():
                with self.assertRaises(DigestBusyError):
                    with other.generation_lock():
                        pass
                raise ValueError("generation failed")
        with other.generation_lock():
            pass

    def test_now_button_is_available_on_single_and_multiple_pages(self):
        for total in (1, 3):
            keyboard = _build_keyboard(0, total)
            self.assertIn("digest_now", [button.callback_data for row in keyboard.inline_keyboard for button in row])
        self.assertIsNone(_build_keyboard(0, 1, include_now=False))


class OnDemandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = DigestStore(Path(self.directory.name) / "digest.sqlite3")
        self.config = Config(tg_chat_id="123", tg_channel_id="-456")
        self.status = SimpleNamespace(edit_text=AsyncMock())
        self.bot = SimpleNamespace(send_message=AsyncMock(side_effect=[self.status, SimpleNamespace(message_id=2)]))

    async def test_button_excludes_both_previous_sections_and_sends_only_to_owner(self):
        self.store.remember("123", [entry("https://example.com/old"), entry("https://example.com/old-mention")])
        generate = Mock(return_value=([entry()], [entry("https://example.com/new-mention")]))
        await handle_digest_now(callback(), self.bot, self.config, generate, self.store)
        generate.assert_called_once_with({"https://example.com/old", "https://example.com/old-mention"})
        self.assertEqual(len(self.store.sent_urls("123")), 4)
        self.assertEqual(self.store.sent_urls("-456"), set())
        self.assertTrue(all(call.kwargs["chat_id"] == "123" for call in self.bot.send_message.call_args_list))
        self.assertIn("Также может быть интересно", self.store.load_pages("123", 2)[0])

    async def test_unauthorized_button_does_not_generate_or_send(self):
        generate = Mock()
        for event in (callback(user_id=456), callback(chat_id=-456)):
            await handle_digest_now(event, self.bot, self.config, generate, self.store)
        generate.assert_not_called()
        self.bot.send_message.assert_not_called()

    async def test_empty_result_reports_no_new_articles(self):
        await handle_digest_now(callback(), self.bot, self.config, Mock(return_value=([], [])), self.store)
        self.status.edit_text.assert_awaited_once_with("Новых статей пока нет. Попробуй позже.")
        self.assertEqual(self.store.sent_urls("123"), set())

    async def test_generation_failure_does_not_record_history(self):
        with patch("digest.bot.logger"):
            await handle_digest_now(callback(), self.bot, self.config, Mock(side_effect=ValueError("invalid JSON")), self.store)
        self.assertEqual(self.store.sent_urls("123"), set())
        self.status.edit_text.assert_awaited_once_with("Не удалось подготовить дайджест. Попробуй ещё раз позже.")

    async def test_failed_telegram_delivery_does_not_record_history(self):
        self.bot.send_message.side_effect = RuntimeError("Telegram unavailable")
        with self.assertRaises(RuntimeError):
            await _send_to_chat(self.bot, self.store, "123", [entry()])
        self.assertEqual(self.store.sent_urls("123"), set())
        self.assertIsNone(self.store.load_pages("123", 2))

    async def test_duplicate_click_is_rejected_while_worker_runs(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def generate(excluded):
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=5)
            return [entry()], []
        task = asyncio.create_task(handle_digest_now(callback(), self.bot, self.config, generate, self.store))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            duplicate = callback()
            other_generate = Mock()
            await handle_digest_now(duplicate, self.bot, self.config, other_generate, self.store)
            other_generate.assert_not_called()
            duplicate.answer.assert_awaited_once_with("Дайджест уже готовится. Дождись завершения.", show_alert=True)
        finally:
            release.set()
            await task

    async def test_pagination_loads_pages_from_persistent_store(self):
        self.store.record_delivery("123", 1, ["first", "second"], [entry()])
        event = callback(data="digest_page:1")
        await handle_page(event, DigestStore(self.store.path))
        self.assertEqual(event.message.edit_text.call_args.kwargs["text"], "second")

    async def test_single_shot_sender_closes_session_without_polling(self):
        fake_bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=3)),
                                   session=SimpleNamespace(close=AsyncMock()))
        with patch("digest.bot.Bot", return_value=fake_bot), patch("digest.bot.DigestStore", return_value=self.store):
            await send_digest([entry()], "test-token", ["123"], [])
        fake_bot.session.close.assert_awaited_once()
        self.assertIn(entry().url, self.store.sent_urls("123"))

    async def test_persistent_bot_routes_start_and_button_without_network(self):
        real_bot = Bot("123:test-token")
        dispatcher = Dispatcher()
        methods = []

        async def request(bot, method, timeout=None):
            methods.append(method)
            if isinstance(method, AnswerCallbackQuery):
                return True
            return Message.model_validate({
                "message_id": len(methods), "date": 0,
                "chat": {"id": 123, "type": "private"}, "text": getattr(method, "text", ""),
            }, context={"bot": bot})

        async def polling(bot, **kwargs):
            await dispatcher.feed_update(bot, Update.model_validate({
                "update_id": 1, "message": {"message_id": 1, "date": 0,
                    "chat": {"id": 123, "type": "private"},
                    "from": {"id": 123, "is_bot": False, "first_name": "Owner"},
                    "text": "/start", "entities": [{"type": "bot_command", "offset": 0, "length": 6}]},
            }))
            await dispatcher.feed_update(bot, Update.model_validate({
                "update_id": 2, "callback_query": {"id": "callback", "chat_instance": "chat",
                    "from": {"id": 123, "is_bot": False, "first_name": "Owner"},
                    "message": {"message_id": 2, "date": 0, "chat": {"id": 123, "type": "private"}},
                    "data": "digest_now"},
            }))

        generate = Mock(return_value=([entry()], []))
        config = Config(tg_bot_token="123:test-token", tg_chat_id="123")
        with patch("digest.bot.Bot", return_value=real_bot), \
             patch("digest.bot.Dispatcher", return_value=dispatcher), \
             patch.object(dispatcher, "start_polling", side_effect=polling), \
             patch.object(real_bot.session, "make_request", side_effect=request):
            await run_bot(config, generate, self.store)
        generate.assert_called_once_with(set())
        self.assertIn(entry().url, self.store.sent_urls("123"))
        welcome = next(method for method in methods if isinstance(method, SendMessage))
        self.assertEqual(welcome.reply_markup.inline_keyboard[-1][0].callback_data, "digest_now")


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        for name, filename in (("ARTICLES_FILE", "articles.json"), ("DIGEST_OUTPUT_FILE", "digest_output.json"),
                               ("LLM_RESPONSE_FILE", "llm_response.txt"), ("LLM_API_RESPONSE_FILE", "llm_response.json")):
            patcher = patch.object(main, name, root / filename)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_all_seen_skips_llm_and_preserves_last_digest(self):
        main.DIGEST_OUTPUT_FILE.write_text("previous digest")
        article = Article("Old", "https://example.com/old", "habr", "", "")
        with patch.object(main, "collect_articles", return_value=[article]), patch.object(main, "get_provider") as provider:
            self.assertEqual(main.generate_digest(Config(), exclude_urls={article.url}), ([], []))
        provider.assert_not_called()
        self.assertEqual(main.DIGEST_OUTPUT_FILE.read_text(), "previous digest")

    def test_parser_excludes_seen_before_fetching_stats(self):
        old = Article("Old", "https://example.com/old/?utm_source=rss", "habr", "", "")
        new = Article("New", "https://example.com/new", "habr", "", "")
        with patch("digest.parser.FEEDS", {"habr": [("ML", "feed")]}), \
             patch("digest.parser.parse_feed", return_value=[old, new]), \
             patch("digest.parser.fetch_habr_stats", return_value=None) as stats, \
             patch("digest.parser.time.sleep"):
            articles = collect_articles(exclude_urls={"https://example.com/old"})
        self.assertEqual([item.url for item in articles], [new.url])
        self.assertEqual(stats.call_args.args[0], new.url)
        self.assertEqual(stats.call_count, 1)

    def test_dry_run_does_not_update_delivery_history(self):
        store = DigestStore(Path(self.directory.name) / "history.sqlite3")
        store.remember("123", [entry("https://example.com/old")])
        with patch.object(main, "DigestStore", return_value=store), \
             patch.object(main.Config, "from_env", return_value=Config(tg_chat_id="123")), \
             patch.object(main, "generate_digest", return_value=([entry()], [])) as generate, \
             patch.object(main, "send_digest") as send, \
             patch("sys.argv", ["main.py", "--dry-run"]):
            main.main()
        self.assertEqual(generate.call_args.args[2], {"https://example.com/old"})
        send.assert_not_called()
        self.assertEqual(store.sent_urls("123"), {"https://example.com/old"})

    def test_generation_filters_seen_unknown_and_duplicate_selections(self):
        old = Article("Old", "https://example.com/old", "habr", "", "")
        new = Article("New", entry().url, "habr", "", "")
        answer = {"articles": [asdict(entry(old.url)), asdict(entry()),
                               asdict(entry("https://example.com/invented")), asdict(entry())]}
        provider = SimpleNamespace(complete=Mock(return_value=json.dumps(answer)), last_response_json=None)
        with patch.object(main, "collect_articles", return_value=[old, new]), patch.object(main, "get_provider", return_value=provider):
            entries, mentions = main.generate_digest(Config(fetch_habr_stats=False), exclude_urls={old.url})
        self.assertEqual([item.url for item in entries], [new.url])
        self.assertEqual(mentions, [])
        self.assertEqual([item["url"] for item in json.loads(provider.complete.call_args.args[1])], [new.url])


if __name__ == "__main__":
    unittest.main()
