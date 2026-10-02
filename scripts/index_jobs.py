"""Index jobs into Qdrant: embed new/changed jobs, or refresh payloads only."""

import argparse

from qdrant_client import QdrantClient

from app.config import Settings
from app.db.session import SessionLocal
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.qdrant import QdrantJobIndex


def main(*, full: bool = False, payload_only: bool = False) -> None:
    settings = Settings()
    # Payload refresh needs no embedding provider, so it works without a Google key.
    embedder = None if payload_only else build_embedding_provider(settings)
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    index = QdrantJobIndex(
        client,
        embedder,
        settings.qdrant_collection,
        settings=settings,
        progress=lambda done, total: print(f"indexed {done}/{total}"),
    )
    with SessionLocal() as session:
        if payload_only:
            result = index.refresh_payloads(session)
            print(
                f"collection={settings.qdrant_collection} refreshed={result.refreshed} "
                "(payload only, no embedding)"
            )
            return
        result = index.index_all(session, full=full)
    print(
        f"collection={settings.qdrant_collection} indexed={result.indexed} "
        f"deleted={result.deleted} "
        f"skipped={result.skipped}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Embed every job again")
    mode.add_argument(
        "--payload-only",
        action="store_true",
        help="Refresh display fields and hashes without embedding (renamed skills, casing)",
    )
    args = parser.parse_args()
    main(full=args.full, payload_only=args.payload_only)
