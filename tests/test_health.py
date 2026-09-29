from app.main import app, get_readiness_qdrant


def test_health(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_checks_database_and_qdrant(client) -> None:
    class FakeQdrant:
        def get_collections(self):
            return []

    app.dependency_overrides[get_readiness_qdrant] = lambda: FakeQdrant()
    try:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "postgres": "ok", "qdrant": "ok"}
    finally:
        app.dependency_overrides.pop(get_readiness_qdrant, None)


def test_ready_returns_503_when_qdrant_fails(client) -> None:
    class BrokenQdrant:
        def get_collections(self):
            raise ConnectionError("offline")

    app.dependency_overrides[get_readiness_qdrant] = lambda: BrokenQdrant()
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"detail": "Dependency unavailable"}
    finally:
        app.dependency_overrides.pop(get_readiness_qdrant, None)
