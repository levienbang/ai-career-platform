import argparse

from qdrant_client import QdrantClient

from app.config import Settings
from app.db.session import SessionLocal
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.qdrant import QdrantJobIndex


def main(*, full: bool = False) -> None:
    settings = Settings()
    embedder = build_embedding_provider(settings)
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    with SessionLocal() as session:
        result = QdrantJobIndex(
            client,
            embedder,
            settings.qdrant_collection,
            settings=settings,
            progress=lambda done, total: print(f"indexed {done}/{total}"),
        ).index_all(session, full=full)
    print(
        f"collection={settings.qdrant_collection} indexed={result.indexed} "
        f"deleted={result.deleted} "
        f"skipped={result.skipped}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally index jobs into Qdrant")
    parser.add_argument("--full", action="store_true", help="Embed every job again")
    main(full=parser.parse_args().full)
