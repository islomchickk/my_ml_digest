import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiogram import Bot
from aiogram.methods import EditMessageText
from aiogram.types import CallbackQuery

from digest.bot import _build_keyboard, handle_digest_now, handle_page
from digest.config import Config
from digest.history import DigestStore
from digest.models import DigestEntry


def entry(index, summary="Summary"):
    return DigestEntry(f"Article {index}", f"https://example.com/{index}", "habr", "Author", [], summary)


def callback(message_id, data, chat_id=123):
    return SimpleNamespace(
        message=SimpleNamespace(message_id=message_id, chat=SimpleNamespace(id=chat_id),
                                edit_text=AsyncMock()),
        data=data, from_user=SimpleNamespace(id=123), answer=AsyncMock(),
    )


class MessagePaginationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "digest.sqlite3"
        self.store = DigestStore(self.path)
        self.store.record_delivery("123", 10, ["Old first", "Old second"], [entry(0)], digest_id="old")

    async def test_new_digest_does_not_edit_source_message_or_replace_its_pages(self):
        source = callback(10, "digest_now")
        status = SimpleNamespace(message_id=20, edit_text=AsyncMock())
        bot = SimpleNamespace(send_message=AsyncMock(side_effect=[status, SimpleNamespace(message_id=21)]))
        generate = Mock(return_value=([entry(1, "New first " * 250), entry(2, "New second " * 250)], []))
        await handle_digest_now(source, bot, Config(tg_chat_id="123"), generate, self.store)
        source.message.edit_text.assert_not_awaited()
        generate.assert_called_once_with({entry(0).url})
        self.assertEqual(self.store.load_pages("123", 10), ["Old first", "Old second"])
        new_pages = self.store.load_pages("123", 21)
        self.assertEqual(len(new_pages), 2)
        keyboard = bot.send_message.call_args_list[1].kwargs["reply_markup"]
        forward = keyboard.inline_keyboard[0][0].callback_data
        token = forward.split(":")[1]
        self.assertNotEqual(token, "old")

        # Exercise both messages against a fresh store, as after a bot restart.
        restarted = DigestStore(self.path)
        old_event = callback(10, "digest_page:old:1")
        await handle_page(old_event, restarted)
        self.assertEqual(old_event.message.edit_text.call_args.kwargs["text"], "Old second")
        new_event = callback(21, forward)
        await handle_page(new_event, restarted)
        self.assertEqual(new_event.message.edit_text.call_args.kwargs["text"], new_pages[1])
        back = new_event.message.edit_text.call_args.kwargs["reply_markup"].inline_keyboard[0][0].callback_data
        self.assertEqual(back, f"digest_page:{token}:0")
        old_back = callback(10, "digest_page:old:0")
        await handle_page(old_back, restarted)
        self.assertEqual(old_back.message.edit_text.call_args.kwargs["text"], "Old first")

    async def test_digest_token_cannot_select_pages_of_another_message_or_chat(self):
        self.store.record_delivery("123", 11, ["New first", "New second"], [entry(1)], digest_id="new")
        for event in (callback(10, "digest_page:new:1"), callback(11, "digest_page:old:1"),
                      callback(10, "digest_page:old:1", chat_id=456)):
            with self.subTest(data=event.data, chat=event.message.chat.id):
                await handle_page(event, self.store)
                event.message.edit_text.assert_not_awaited()
                event.answer.assert_awaited_once_with("Дайджест недоступен")

    async def test_duplicate_delivery_does_not_overwrite_original_snapshot(self):
        self.store.record_delivery("123", 10, ["Replacement"], [entry(1)], digest_id="replacement")
        self.assertEqual(self.store.load_pages("123", 10, digest_id="old"), ["Old first", "Old second"])
        self.assertIsNone(self.store.load_pages("123", 10, digest_id="replacement"))

    async def test_legacy_buttons_still_use_only_clicked_message(self):
        self.store.record_delivery("123", 11, ["Legacy first", "Legacy second"], [entry(1)])
        for message_id, expected in ((10, "Old second"), (11, "Legacy second")):
            event = callback(message_id, "digest_page:1")
            await handle_page(event, self.store)
            self.assertEqual(event.message.edit_text.call_args.kwargs["text"], expected)

    async def test_aiogram_edits_only_the_message_containing_clicked_keyboard(self):
        self.store.record_delivery("123", 11, ["New first", "New second"], [entry(1)], digest_id="new")
        bot = Bot("123:test-token")
        self.addAsyncCleanup(bot.session.close)
        with patch.object(bot.session, "make_request", new_callable=AsyncMock, return_value=True) as request:
            for message_id, token in ((10, "old"), (11, "new"), (10, "old")):
                event = CallbackQuery.model_validate({
                    "id": f"event-{message_id}", "chat_instance": "chat",
                    "from": {"id": 123, "is_bot": False, "first_name": "Owner"},
                    "message": {"message_id": message_id, "date": 1,
                                "chat": {"id": 123, "type": "private"}},
                    "data": f"digest_page:{token}:1",
                }, context={"bot": bot})
                await handle_page(event, self.store)
        edits = [call.args[1] for call in request.call_args_list if isinstance(call.args[1], EditMessageText)]
        self.assertEqual([method.message_id for method in edits], [10, 11, 10])
        self.assertEqual([method.chat_id for method in edits], [123, 123, 123])
        self.assertEqual([method.text for method in edits], ["Old second", "New second", "Old second"])

    async def test_invalid_page_callbacks_do_not_edit_messages(self):
        for data in ("digest_page:old:not-an-integer", "digest_page::1", "digest_page:old:1:extra",
                     "digest_page:old:-1", "digest_page:old:100", None):
            event = callback(10, data)
            with self.subTest(data=data):
                await handle_page(event, self.store)
                event.message.edit_text.assert_not_awaited()

    def test_keyboard_token_fits_telegram_callback_limit(self):
        keyboard = _build_keyboard(100, 1000, digest_id="a" * 32)
        for row in keyboard.inline_keyboard:
            for button in row:
                self.assertLessEqual(len(button.callback_data.encode()), 64)


if __name__ == "__main__":
    unittest.main()
