"""Отправка дайджеста в Telegram через aiogram 3 с пагинацией."""

from collections import OrderedDict

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from digest.models import DigestEntry

# {message_id: pages_list} — хранилище страниц для callback'ов
_pages_store: dict[int, list[str]] = {}


def _format_entry(entry: DigestEntry) -> str:
    """Форматирует одну статью как буллет."""
    return f'• <a href="{entry.url}">{entry.title}</a> — {entry.summary}'


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
            cat_header = f"\n\n<b>{cat}</b>" if i == 0 else ""
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


def _build_keyboard(page: int, total: int) -> InlineKeyboardMarkup | None:
    """Создаёт inline-клавиатуру с кнопками навигации."""
    if total <= 1:
        return None

    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"digest_page:{page - 1}"))
    if page < total - 1:
        buttons.append(InlineKeyboardButton(text="Вперёд ➡️", callback_data=f"digest_page:{page + 1}"))

    return InlineKeyboardMarkup(inline_keyboard=[buttons])


async def send_digest(
    entries: list[DigestEntry],
    bot_token: str,
    chat_ids: list[str],
    mentions: list[DigestEntry] | None = None,
) -> None:
    """Отправляет дайджест и запускает polling для обработки кнопок пагинации."""
    bot = Bot(token=bot_token)
    dp = Dispatcher()

    pages = _build_pages(entries, mentions)
    keyboard = _build_keyboard(0, len(pages))

    # Отправляем первую страницу во все чаты
    for chat_id in chat_ids:
        msg = await bot.send_message(
            chat_id=chat_id,
            text=pages[0],
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=keyboard,
        )
        _pages_store[msg.message_id] = pages

    # Регистрируем callback handler
    @dp.callback_query(F.data.startswith("digest_page:"))
    async def on_page_callback(callback: CallbackQuery) -> None:
        page = int(callback.data.split(":")[1])
        stored_pages = _pages_store.get(callback.message.message_id)
        if not stored_pages:
            await callback.answer("Дайджест устарел")
            return

        total = len(stored_pages)
        if page < 0 or page >= total:
            await callback.answer()
            return

        await callback.message.edit_text(
            text=stored_pages[page],
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=_build_keyboard(page, total),
        )
        await callback.answer()

    # Запускаем polling для обработки callback'ов
    print("Bot is listening for pagination callbacks (Ctrl+C to stop)...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
