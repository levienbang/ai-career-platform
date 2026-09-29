import json
from pathlib import Path

from evaluation.evaluate_milestone7 import FixtureEmbeddingProvider, FixtureRouter

ROOT = Path(__file__).parents[1]


def test_ground_truth_manifest_has_39_prelabelled_cases():
    manifest = json.loads((ROOT / "evaluation/milestone7_cases.json").read_text())
    retrieval = json.loads((ROOT / "evaluation/retrieval_cases.json").read_text())
    routing = json.loads((ROOT / "evaluation/agent_cases.json").read_text())
    assert [
        len(retrieval),
        len(routing),
        len(manifest["sql"]),
        len(manifest["cv"]),
        len(manifest["error"]),
    ] == [12, 12, 5, 5, 5]
    assert all("expected_count" in case for case in manifest["sql"])
    assert all("expected_fit" in case for case in manifest["cv"])
    assert all("expected" in case for case in manifest["error"])


def test_sql_ground_truth_matches_seed_fixture():
    manifest = json.loads((ROOT / "evaluation/milestone7_cases.json").read_text())
    seed = json.loads((ROOT / "data/seed_data.json").read_text())
    jobs = [
        job for job in seed["jobs"] if job["source_url"].startswith("https://example.com/eval/")
    ]
    expected = [
        len(jobs),
        sum("Docker" in job["skills"] for job in jobs),
        sum("FastAPI" in job["skills"] for job in jobs),
        sum("PostgreSQL" in job["skills"] for job in jobs),
        sum(job["employment_type"] == "Internship" for job in jobs),
    ]
    assert [case["expected_count"] for case in manifest["sql"]] == expected


def test_fake_provider_and_router_are_deterministic():
    embedder = FixtureEmbeddingProvider()
    assert embedder.embed_query("Python Docker") == embedder.embed_query("Python Docker")
    assert FixtureRouter().decide("Tìm job và đếm số job").route == "both"
