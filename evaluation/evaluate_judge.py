"""Grade saved retrieval rankings with an LLM judge (DeepSeek) instead of silver labels.

The pool is, per case in real_cases.json (retrieval + retrieval_paraphrase), the union
of the top-10 job IDs every method produced in real_results.json. Each (query, job)
pair is graded 0/1/2 from the job's search document; judgments are cached in
judge_cache.json, so reruns only pay for new or changed pairs.

Dry-run by default: prints the planned DeepSeek calls and exits (no network, no DB).
  --run              grade uncached pairs (paid), then report and write the human sample
  --report           recompute judge_results.json from the cache only (free, no DB)
  --write-human-sample   (re)write the blind human-review sample (free, needs the DB)
  --human-agreement  judge-vs-human agreement from the filled-in sample (free)
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.judge import (
    JudgmentCache,
    build_human_review,
    human_agreement,
    human_review_started,
    judge_pool,
    judged_precision_at_k,
    ndcg_at_k,
    planned_calls,
    sample_pairs,
    silver_agreement,
    truncate_document,
)

HERE = Path(__file__).parent
CASES_FILE = HERE / "real_cases.json"
RETRIEVAL_RESULTS_FILE = HERE / "real_results.json"
CACHE_FILE = HERE / "judge_cache.json"
JUDGE_RESULTS_FILE = HERE / "judge_results.json"
HUMAN_REVIEW_FILE = HERE / "judge_human_review.json"

SECTIONS = ("retrieval", "retrieval_paraphrase")
# Silver labels were frozen when the corpus held jobs 1..941; later jobs are unlabelled,
# so they are left out of judge-vs-silver agreement instead of counted as non-relevant.
LABELLED_MAX_JOB_ID = 941
DEFAULT_BATCH_SIZE = 8
HUMAN_SAMPLE_SIZE = 30
HUMAN_SAMPLE_SEED = 20261118


def method_rankings(case_result: dict[str, Any]) -> dict[str, list[int]]:
    return {
        name: value["top10"]
        for name, value in case_result.items()
        if isinstance(value, dict) and "top10" in value
    }


def build_pool(results: dict[str, Any], sections=SECTIONS) -> dict[str, list[int]]:
    """{query: sorted union of every method's top-10 job IDs}."""
    pool: dict[str, set[int]] = {}
    for section in sections:
        for case in results.get(section, {}).get("cases", []):
            ids = pool.setdefault(case["query"], set())
            for ranking in method_rankings(case).values():
                ids.update(ranking[:10])
    return {query: sorted(ids) for query, ids in pool.items()}


def build_report(
    cases: dict[str, Any],
    results: dict[str, Any],
    cache: JudgmentCache,
    *,
    labelled_max_job_id: int = LABELLED_MAX_JOB_ID,
    sections=SECTIONS,
) -> dict[str, Any]:
    """Judged P@5 / nDCG@10 per method, and judge-vs-silver agreement, from the cache."""
    report: dict[str, Any] = {"labelled_max_job_id": labelled_max_job_id}
    silver_pairs: dict[tuple[str, int], tuple[int, bool]] = {}
    unlabelled_pairs: set[tuple[str, int]] = set()
    for section in sections:
        relevant_by_id = {case["id"]: set(case["relevant_job_ids"]) for case in cases[section]}
        scores: dict[str, dict[str, list[float]]] = {}
        incomplete: list[str] = []
        case_rows = []
        for case in results.get(section, {}).get("cases", []):
            query, rankings = case["query"], method_rankings(case)
            pool_ids = sorted({job_id for ids in rankings.values() for job_id in ids[:10]})
            grades = {job_id: cache.grade(query, job_id) for job_id in pool_ids}
            if any(grade is None for grade in grades.values()):
                incomplete.append(case["id"])
                continue
            relevant = relevant_by_id[case["id"]]
            for job_id, grade in grades.items():
                if job_id > labelled_max_job_id:
                    unlabelled_pairs.add((query, job_id))
                    continue
                silver_pairs[(query, job_id)] = (grade, job_id in relevant)
            row: dict[str, Any] = {"id": case["id"], "query": query}
            for method, ids in rankings.items():
                ranked = [grades[job_id] for job_id in ids[:10]]
                method_scores = {
                    "judged_precision_at_5": judged_precision_at_k(ranked, 5, threshold=2),
                    "judged_precision_at_5_lenient": judged_precision_at_k(ranked, 5, threshold=1),
                    "ndcg_at_10": ndcg_at_k(ranked, list(grades.values()), 10),
                }
                for metric, value in method_scores.items():
                    scores.setdefault(method, {}).setdefault(metric, []).append(value)
                row[method] = {
                    "grades_top10": ranked,
                    **{k: round(v, 3) for k, v in method_scores.items()},
                }
            case_rows.append(row)
        summary = {
            method: {
                **{metric: round(mean(values), 3) for metric, values in metrics.items()},
                "silver_precision_at_5": results[section].get(method, {}).get("precision_at_5"),
            }
            for method, metrics in scores.items()
        }
        report[section] = {
            **summary,
            "judged_cases": len(case_rows),
            "incomplete_cases": incomplete,
            "cases": case_rows,
        }
    pairs = list(silver_pairs.values())
    report["silver_agreement"] = {
        "unlabelled_pairs_skipped": len(unlabelled_pairs),
        "relevant_if_grade_ge_1": silver_agreement(pairs, threshold=1),
        "relevant_if_grade_ge_2": silver_agreement(pairs, threshold=2),
    }
    return report


def load_documents(job_ids: list[int]) -> dict[int, str]:
    """Search documents exactly as indexed, truncated for the judge prompt."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.db.models import Job, JobSkill
    from app.db.session import SessionLocal
    from app.retrieval.documents import build_search_document

    with SessionLocal() as session:
        jobs = session.scalars(
            select(Job)
            .where(Job.id.in_(job_ids))
            .options(
                selectinload(Job.company), selectinload(Job.skills).selectinload(JobSkill.skill)
            )
        ).all()
        documents = {job.id: truncate_document(build_search_document(job).text) for job in jobs}
    missing = sorted(set(job_ids) - set(documents))
    if missing:
        raise SystemExit(f"Jobs missing from the database: {missing}")
    return documents


def write_human_sample(pool: dict[str, list[int]], documents: dict[int, str], cases) -> int:
    if human_review_started(HUMAN_REVIEW_FILE):
        print(f"{HUMAN_REVIEW_FILE.name} already has human grades; not overwriting it.")
        return 0
    pairs = [(query, job_id) for query, ids in pool.items() for job_id in ids]
    sampled = sample_pairs(pairs, HUMAN_SAMPLE_SIZE, HUMAN_SAMPLE_SEED)
    case_ids = {case["query"]: case["id"] for section in SECTIONS for case in cases[section]}
    review = build_human_review(sampled, documents, case_ids, seed=HUMAN_SAMPLE_SEED)
    HUMAN_REVIEW_FILE.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(sampled)} pairs to {HUMAN_REVIEW_FILE.name}")
    return len(sampled)


def save_report(cases, results, cache) -> None:
    report = build_report(cases, results, cache)
    JUDGE_RESULTS_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {
        section: {k: v for k, v in report[section].items() if k != "cases"} for section in SECTIONS
    }
    summary["silver_agreement"] = report["silver_agreement"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {JUDGE_RESULTS_FILE.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Make the paid DeepSeek calls")
    mode.add_argument("--report", action="store_true", help="Report from the cache only")
    mode.add_argument("--write-human-sample", action="store_true")
    mode.add_argument("--human-agreement", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    results = json.loads(RETRIEVAL_RESULTS_FILE.read_text(encoding="utf-8"))
    cache = JudgmentCache(CACHE_FILE)
    pool = build_pool(results)
    pairs = sum(len(ids) for ids in pool.values())

    if args.human_agreement:
        review = json.loads(HUMAN_REVIEW_FILE.read_text(encoding="utf-8"))
        print(json.dumps(human_agreement(review, cache), indent=2))
        return
    if args.report:
        if not cache.entries:
            raise SystemExit(f"{CACHE_FILE.name} has no judgments yet; run with --run first.")
        save_report(cases, results, cache)
        return

    calls = planned_calls(cache, pool, args.batch_size)
    print(
        f"Pool: {len(pool)} queries, {pairs} (query, job) pairs. "
        f"Planned DeepSeek calls (batch size {args.batch_size}, before retries): {calls}"
    )
    if not (args.run or args.write_human_sample):
        print("Dry-run only; pass --run to grade (paid) or --report to use the cache.")
        return

    documents = load_documents(sorted({job_id for ids in pool.values() for job_id in ids}))
    if args.write_human_sample:
        write_human_sample(pool, documents, cases)
        return

    from app.config import Settings
    from evaluation.judge import LangChainRelevanceJudge

    calls = planned_calls(cache, pool, args.batch_size, documents)
    print(f"Calls after checking document hashes: {calls}")
    judge = LangChainRelevanceJudge(Settings())
    stats = judge_pool(judge, cache, pool, documents, batch_size=args.batch_size)
    print(f"DeepSeek calls made: {stats.calls}; pairs graded: {stats.judged_pairs}")
    for failure in stats.failed_batches:
        print(f"Failed batch for {failure['query']!r}: {failure['error']}")
    save_report(cases, results, cache)
    if not HUMAN_REVIEW_FILE.exists():
        write_human_sample(pool, documents, cases)


if __name__ == "__main__":
    main()
