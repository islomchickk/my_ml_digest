FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY digest/ digest/
COPY main.py .

CMD ["uv", "run", "python", "main.py"]
