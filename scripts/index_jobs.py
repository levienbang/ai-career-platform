from qdrant_client import QdrantClient

from app.config import Settings
from app.db.session import SessionLocal
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.qdrant import QdrantJobIndex


def main() -> None:
    settings = Settings()
    embedder = build_embedding_provider(settings)
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    with SessionLocal() as session:
        result = QdrantJobIndex(client, embedder, settings.qdrant_collection).index_all(session)
    print(
        f"collection={settings.qdrant_collection} indexed={result.indexed} deleted={result.deleted}"
    )


if __name__ == "__main__":
    main()
