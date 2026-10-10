"""Back up or restore the job vectors through Qdrant's snapshot API.

The snapshot is streamed through stdout/stdin so it works from inside the API
container without sharing a volume:

    docker compose exec -T api python -m scripts.qdrant_snapshot save > jobs.snapshot
    docker compose exec -T api python -m scripts.qdrant_snapshot restore < jobs.snapshot

Restoring avoids re-embedding every job (paid Google calls) after a move or crash.
"""

import argparse
import sys

import httpx

from app.config import Settings


def save(client: httpx.Client, collection: str) -> None:
    response = client.post(f"/collections/{collection}/snapshots", params={"wait": "true"})
    response.raise_for_status()
    name = response.json()["result"]["name"]
    try:
        with client.stream("GET", f"/collections/{collection}/snapshots/{name}") as download:
            download.raise_for_status()
            for chunk in download.iter_bytes():
                sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    finally:
        # The copy now lives outside the server; do not let snapshots fill its disk.
        client.delete(f"/collections/{collection}/snapshots/{name}")


def restore(client: httpx.Client, collection: str) -> None:
    data = sys.stdin.buffer.read()
    if not data:
        raise SystemExit("restore: empty input, pipe a snapshot file on stdin")
    response = client.post(
        f"/collections/{collection}/snapshots/upload",
        params={"priority": "snapshot", "wait": "true"},
        files={"snapshot": (f"{collection}.snapshot", data)},
    )
    response.raise_for_status()
    count = client.post(f"/collections/{collection}/points/count", json={"exact": True})
    count.raise_for_status()
    print(f"restored collection={collection} points={count.json()['result']['count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["save", "restore"])
    action = parser.parse_args().action
    settings = Settings()
    with httpx.Client(base_url=settings.qdrant_url, timeout=600) as client:
        (save if action == "save" else restore)(client, settings.qdrant_collection)


if __name__ == "__main__":
    main()
