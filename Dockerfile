FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY migrations ./migrations
COPY scripts ./scripts
COPY data ./data
COPY evaluation ./evaluation
COPY alembic.ini ./
ARG INSTALL_TARGET=.
RUN pip install --no-cache-dir "${INSTALL_TARGET}"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
