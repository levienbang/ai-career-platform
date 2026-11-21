"""Chat history in a local SQLite file through Chainlit's SQLAlchemy data layer.

Chainlit does not create its tables, so this module does on first use. Uploaded
files (CV PDFs) are not stored: no storage provider is configured, and only the
messages are kept. The file is gitignored; delete it to clear all history.
"""

import os
import sqlite3
from pathlib import Path

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

DB_PATH = Path(os.getenv("CHAT_HISTORY_DB", Path(__file__).parent / ".chats.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT REFERENCES users("id") ON DELETE CASCADE,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL REFERENCES threads("id") ON DELETE CASCADE,
    "parentId" TEXT,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "modes" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER,
    "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN,
    "icon" TEXT
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT REFERENCES threads("id") ON DELETE CASCADE,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    "autoPlay" BOOLEAN,
    "playerConfig" TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL REFERENCES threads("id") ON DELETE CASCADE,
    "value" INTEGER NOT NULL,
    "comment" TEXT
);
"""


def ensure_schema(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)


def build_data_layer(path: Path = DB_PATH) -> SQLAlchemyDataLayer:
    ensure_schema(path)
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{path}")
