FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
RUN uv run python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY digest/ digest/
COPY main.py .

CMD ["uv", "run", "python", "main.py"]
