# Digest

Еженедельный дайджест статей по Data Science, Machine Learning и AI с автоматической отправкой в Telegram.

## Что делает

Каждый понедельник утром бот:

1. **Собирает статьи** из RSS-лент Хабра, Medium и Towards Data Science по темам: ML, AI, NLP, Data Engineering, Data Science, Python, LLM, RAG, MLOps.
2. **Обогащает статистикой** — для Хабра подтягивает рейтинг, просмотры, комментарии и закладки.
3. **Фильтрует через LLM** — отбирает 10 лучших статей и пишет краткое саммари на русском.
4. **Отправляет в Telegram** — дайджест с пагинацией и inline-кнопками.

## Источники

| Источник | Темы |
|----------|------|
| Хабр | ML, AI, NLP, Data Engineering, Data Science, Python |
| Medium | ML, LLM, RAG, AI, MLOps, Data Science |
| Towards Data Science | Все публикации |

## Поддерживаемые LLM-провайдеры

- Claude
- OpenAI
- Gemini
- OpenRouter
- NeuralDeep (по умолчанию, OpenAI-совместимый API)

## Быстрый старт

```bash
# Клонируй репозиторий
git clone <repo-url>
cd digest

# Создай .env из примера и заполни ключи
cp .env.example .env

# Запусти вручную
uv run python main.py
```

### Флаги

```
--llm <provider>   LLM-провайдер: claude, openai, gemini, openrouter, neuraldeep
--no-stats         Не собирать статистику Хабра
--dry-run          Не отправлять в Telegram
--no-parse         Пропустить парсинг, загрузить из articles.json
--test-send        Отправить digest_output.json в TG_CHAT_ID (без парсинга и LLM)
```

## Docker

```bash
# Заполни .env, затем:
docker compose up -d
```

Дайджест отправляется автоматически **каждый понедельник в 09:00** через cron-планировщик Ofelia. Для ручного запуска:

```bash
docker compose exec digest uv run python main.py
```

## Переменные окружения

```env
# Telegram
TG_BOT_TOKEN=         # токен бота
TG_CHAT_ID=           # ID чата для отправки
TG_CHANNEL_ID=        # ID канала (опционально)

# LLM
LLM_PROVIDER=neuraldeep  # claude | openai | gemini | openrouter | neuraldeep
LLM_MODEL=qwen3.6-unlim  # модель (опционально, используется дефолтная)

# API-ключи (заполни только нужный)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
NEURALDEEP_API_KEY=

# Парсер
FETCH_HABR_STATS=true
```
