FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    CHAT_HISTORY_DB=/data/chats.db

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /chat

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app.py api.py context.py history.py render.py chainlit.md ./
COPY .chainlit/config.toml .chainlit/config.toml
COPY public public
# Chainlit writes translations and uploads under the app folder; history goes to /data.
RUN useradd --create-home chat && mkdir -p /data && chown -R chat /chat /data
USER chat

EXPOSE 8000
CMD ["/opt/venv/bin/chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
