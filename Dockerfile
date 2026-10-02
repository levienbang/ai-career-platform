FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
ARG INSTALL_TARGET=.
# Build metadata alone is enough to resolve dependencies; source changes preserve this layer.
RUN pip install --no-cache-dir "${INSTALL_TARGET}"

COPY app ./app
COPY migrations ./migrations
COPY scripts ./scripts
COPY data ./data
COPY evaluation ./evaluation
COPY alembic.ini ./
RUN pip install --no-cache-dir --no-deps "${INSTALL_TARGET}"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
