"""Отправка дайджеста в Telegram через aiogram 3 с пагинацией."""

from collections import OrderedDict
import asyncio
import logging
from html import escape
from typing import Callable

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from digest.models import DigestEntry
from digest.config import Config
from digest.history import DigestStore, DigestBusyError

logger = logging.getLogger(__name__)


def _format_entry(entry: DigestEntry) -> str:
    """Форматирует одну статью как буллет."""
    return f'• <a href="{escape(entry.url, quote=True)}">{escape(entry.title)}</a> — {escape(entry.summary)}'


def _build_pages(
    entries: list[DigestEntry],
    mentions: list[DigestEntry] | None = None,
    max_len: int = 4000,
) -> list[str]:
    """Разбивает дайджест на страницы по границам статей.

    Каждая страница содержит заголовок и целые статьи, сгруппированные по категориям.
    Статья никогда не разрывается между страницами.
    """
    header = "<b>Weekly Digest</b>"

    # Группируем статьи по категориям (сохраняя порядок появления)
    categories: OrderedDict[str, list[DigestEntry]] = OrderedDict()
    for entry in entries:
        cat = entry.category or "Прочее"
        categories.setdefault(cat, []).append(entry)

    # Готовим блоки: (category_header, entry_text) для каждой статьи
    blocks: list[tuple[str, str]] = []
    for cat, cat_entries in categories.items():
        for i, entry in enumerate(cat_entries):
            # Заголовок категории только перед первой статьёй в ней
            cat_header = f"\n\n<b>{escape(cat)}</b>" if i == 0 else ""
            blocks.append((cat_header, _format_entry(entry)))

    # Mentions как отдельные блоки
    mention_blocks: list[tuple[str, str]] = []
    if mentions:
        for i, m in enumerate(mentions):
            cat_header = "\n\n<b>Также может быть интересно:</b>" if i == 0 else ""
            mention_blocks.append((cat_header, _format_entry(m)))

    all_blocks = blocks + mention_blocks

    # Собираем страницы
    pages: list[str] = []
    current = header

    for cat_header, entry_text in all_blocks:
        block = cat_header + "\n" + entry_text if cat_header else "\n\n" + entry_text

        if len(current) + len(block) > max_len and current != header:
            pages.append(current)
            # Новая страница: заголовок + (если это продолжение категории, без заголовка)
            current = header + block
        else:
            current += block

    if current and current != header:
        pages.append(current)

    # Добавляем номера страниц если их больше 1
    if len(pages) > 1:
        total = len(pages)
        pages = [
            page + f"\n\n<i>Стр. {i + 1}/{total}</i>"
            for i, page in enumerate(pages)
        ]

    return pages if pages else [header]


def _build_keyboard(page: int, total: int, include_now: bool = True) -> InlineKeyboardMarkup | None:
    """Создаёт inline-клавиатуру с кнопками навигации."""
    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"digest_page:{page - 1}"))
    if page < total - 1:
        buttons.append(InlineKeyboardButton(text="Вперёд ➡️", callback_data=f"digest_page:{page + 1}"))

    rows = [buttons] if buttons else []
    if include_now:
        rows.append([InlineKeyboardButton(text="📰 Дайджест сейчас", callback_data="digest_now")])
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


async def _send_to_chat(
    bot: Bot, store: DigestStore, chat_id: str,
    entries: list[DigestEntry], mentions: list[DigestEntry] | None = None,
) -> None:
    pages = _build_pages(entries, mentions)
    message = await bot.send_message(
        chat_id=chat_id, text=pages[0], parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=_build_keyboard(0, len(pages), include_now=int(chat_id) > 0),
    )
    # Persist only after Telegram accepts the message. The whole digest,
    # including the recommendations on later pages, is considered delivered.
    store.record_delivery(chat_id, message.message_id, pages, entries + (mentions or []))


async def send_digest(
    entries: list[DigestEntry],
    bot_token: str,
    chat_ids: list[str],
    mentions: list[DigestEntry] | None = None,
) -> None:
    """Single-shot delivery; the persistent --bot process handles all buttons."""
    bot = Bot(token=bot_token)
    try:
        store = DigestStore()
        for chat_id in chat_ids:
            await _send_to_chat(bot, store, chat_id, entries, mentions)
    finally:
        await bot.session.close()


def _is_owner(chat_id: int, user_id: int, config: Config) -> bool:
    return str(chat_id) == config.tg_chat_id and str(user_id) == config.tg_chat_id


async def handle_page(callback: CallbackQuery, store: DigestStore) -> None:
    if callback.message is None:
        await callback.answer()
        return
    pages = store.load_pages(str(callback.message.chat.id), callback.message.message_id)
    if not pages:
        await callback.answer("Дайджест устарел")
        return
    try:
        page = int(callback.data.split(":", 1)[1])
    except (ValueError, IndexError, AttributeError):
        await callback.answer()
        return
    await callback.answer()
    if 0 <= page < len(pages):
        await callback.message.edit_text(
            text=pages[page], parse_mode=ParseMode.HTML, disable_web_page_preview=True,
            reply_markup=_build_keyboard(page, len(pages), include_now=callback.message.chat.id > 0),
        )


async def handle_digest_now(
    callback: CallbackQuery, bot: Bot, config: Config,
    generate: Callable[[set[str]], tuple[list[DigestEntry], list[DigestEntry]]],
    store: DigestStore,
) -> None:
    if callback.message is None or not _is_owner(callback.message.chat.id, callback.from_user.id, config):
        await callback.answer("Кнопка доступна только владельцу в личном чате", show_alert=True)
        return
    chat_id = str(callback.message.chat.id)
    status = None
    try:
        with store.generation_lock():
            await callback.answer("Собираю новый дайджест")
            status = await bot.send_message(chat_id=chat_id, text="Собираю статьи и готовлю дайджест…")
            # RSS/statistics and the LLM SDK are synchronous. Keep polling and
            # pagination responsive while the generation runs in a worker.
            task = asyncio.create_task(asyncio.to_thread(generate, store.sent_urls(chat_id)))
            try:
                entries, mentions = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Don't release the cross-process lock while the worker is
                # still writing its output files during shutdown.
                await task
                raise
            if not entries:
                await status.edit_text("Новых статей пока нет. Попробуй позже.")
                return
            await _send_to_chat(bot, store, chat_id, entries, mentions)
            await status.edit_text("Новый дайджест готов.")
    except DigestBusyError:
        await callback.answer("Дайджест уже готовится. Дождись завершения.", show_alert=True)
    except Exception:
        logger.exception("On-demand digest failed")
        if status:
            await status.edit_text("Не удалось подготовить дайджест. Попробуй ещё раз позже.")
        else:
            await callback.answer("Не удалось запустить дайджест", show_alert=True)


async def run_bot(
    config: Config,
    generate: Callable[[set[str]], tuple[list[DigestEntry], list[DigestEntry]]],
    store: DigestStore,
) -> None:
    if not config.tg_bot_token or not config.tg_chat_id or int(config.tg_chat_id) <= 0:
        raise ValueError("--bot requires TG_BOT_TOKEN and a personal TG_CHAT_ID")
    bot = Bot(token=config.tg_bot_token)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        if message.from_user and _is_owner(message.chat.id, message.from_user.id, config):
            await message.answer(
                "Нажми «Дайджест сейчас», чтобы получить новые статьи без повторов.",
                reply_markup=_build_keyboard(0, 1),
            )

    @dp.callback_query(F.data == "digest_now")
    async def on_now(callback: CallbackQuery) -> None:
        await handle_digest_now(callback, bot, config, generate, store)

    @dp.callback_query(F.data.startswith("digest_page:"))
    async def on_page(callback: CallbackQuery) -> None:
        await handle_page(callback, store)

    print("Bot is listening for digest requests and pagination callbacks…", flush=True)
    try:
        await dp.start_polling(bot, close_bot_session=False)
    finally:
        await bot.session.close()
