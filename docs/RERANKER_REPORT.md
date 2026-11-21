# Reranker comparison (Milestone 15, 2026-11-21)

Which reranker should reorder the hybrid (BM25 + dense, RRF) candidates before the top 5 are returned?

## Setup

- **Queries:** the 10 paraphrase queries in `evaluation/real_cases.json` (`retrieval_paraphrase`). They describe a job without naming the technology, for example *"lập trình ứng dụng cho điện thoại chạy hệ điều hành của Google"* for Android.
- **Candidates:** the hybrid top 10 for each query, computed once and saved in `evaluation/rerank_pools.json`. Every reranker sees the same pool, so building the pool cost only 10 embedding calls in total.
- **Grades:** Claude graded every job that any system put in its top 5, on a 0–2 scale: 2 is relevant, 1 is partly relevant, 0 is not relevant. The grades are in `evaluation/manual_grades_paraphrase.json`.
- **Metrics:** nDCG@5 with gain 2^g − 1, P@5 counting only grade 2, and lenient P@5 counting grade 1 or higher. Latency is the median wall time to rerank 10 candidates.
- **Hardware:** MacBook M1 with 16 GB RAM. The local models run on the MPS GPU.
- **Query subset:** the main table uses 8 queries. On par-02 and par-05 DeepSeek returned missing or duplicate IDs, which raised `RerankerResponseError`, so those two queries are left out to keep the comparison fair.

## Results (10 candidates, 8 queries)

| Reranker | Type | nDCG@5 | P@5 (=2) | P@5 (≥1) | Latency | Memory |
|---|---|---|---|---|---|---|
| none (hybrid order) | – | 0.605 | 0.45 | 0.78 | – | – |
| `BAAI/bge-reranker-v2-m3` | cross-encoder, 0.57B | 0.631 | 0.47 | 0.82 | 2.3 s | ~1 GB |
| `Qwen/Qwen3-Reranker-0.6B` | decoder yes/no logit | 0.700 | 0.53 | 0.82 | 8.3 s | 3.4 GB GPU |
| `Qwen/Qwen3-Reranker-4B` | decoder yes/no logit | 0.799 | 0.65 | 0.85 | 32.4 s | 9.4 GB GPU, heavy swap |
| DeepSeek `deepseek-v4-flash` | listwise LLM (API) | **0.864** | **0.72** | **0.88** | 1.6 s | – |

On all 10 queries the nDCG@5 scores are:

| Reranker | nDCG@5 (10 queries) |
|---|---|
| hybrid | 0.666 |
| bge | 0.681 |
| Qwen 0.6B | 0.745 |
| Qwen 4B | 0.828 |
| DeepSeek | none: it failed on 2 of the 10 queries |

### 10 vs 20 candidates

| Reranker | nDCG@5, 10 candidates | nDCG@5, 20 candidates | Latency, 10 → 20 | Silver P@5, 10 → 20 |
|---|---|---|---|---|
| bge | 0.631 | 0.680 | 2.3 s → 4.4 s | 0.78 → 0.78 |
| DeepSeek | – | – | tokens double | 0.93 → 0.97 |

Twenty candidates help a little, mostly on par-08: the right Android jobs sit at hybrid ranks 11, 13 and 17, so no reranker can see them with 10 candidates. We kept 10 (`RERANKER_MAX_CANDIDATES=10`) because the gain is small and the cost doubles.

## Findings

1. **A reranker can only reorder what it receives.** On par-08 every reranker fails for the same reason: the relevant jobs are not in the top 10. Candidate recall sets the ceiling.
2. **Larger local models are better but too slow on an M1.**
   - Qwen 4B reaches 92% of DeepSeek's nDCG@5 with no API cost and no invalid output.
   - Each query needs about 9k prompt tokens × 4B parameters ≈ 70 TFLOP. The M1 GPU delivers 2–3 TFLOPS, so a query takes about 30 s.
   - On a server GPU (L4/A100) the same model would take about 1–2 s.
3. **Listwise beats pointwise here, but output can be invalid.**
   - DeepSeek sees all 10 jobs at once and ranks best.
   - In 2 of 10 queries it returned an invalid ID list, which became a 503.
   - The pointwise models always return a valid score for each job.
4. **bge barely beats the hybrid order** on these queries (+0.03 nDCG@5).

## Decision

- **Keep DeepSeek as the main reranker.** It is the best and fastest option.
- **Fall back to the local bge cross-encoder (done, `RERANKER_FALLBACK=cross_encoder`)** on invalid IDs, timeouts and provider errors, then to hybrid order if bge is unavailable, so the user sees a ranked list instead of a 503. The response field `reranker` says which one ran. A missing API key is a configuration error and still returns 503.
- **Keep bge as the local-only option** (`RERANKER_PROVIDER=cross_encoder`).
- **Do not ship Qwen.**
  - The models were deleted from the Hugging Face cache, and the one-off scoring scripts were removed.
  - The raw rankings stay in `evaluation/qwen_0.6b.json` and `evaluation/qwen_4b.json`.
  - Reconsider Qwen3-Reranker-4B if the API is served from a GPU host.
- **Keep 10 candidates.**

## Open questions

- **Candidate source.** Dense alone scores silver P@5 0.88 on paraphrase queries against 0.74 for hybrid: BM25 noise pushes relevant jobs out of the top 10. Two cheap alternatives are worth testing on the saved pools:
  - the dense top 10 plus the BM25 top 5;
  - a lower BM25 weight in RRF.
  The current test set has no exact-term queries (company names, `C#`, versions), so it cannot yet show what BM25 contributes. Add such queries before dropping BM25.
- **Sample size.** With 8–10 queries, differences under about 0.05 nDCG are within noise.

## Notes on the Qwen run

- Qwen3-Reranker scores a job as P("yes") at the last position of a fixed judge prompt.
- Computing only the last-token logits (`logits_to_keep=1`) cut peak memory from 12.5 GB to 3.4 GB for 0.6B, with identical rankings.
- Documents were truncated to about 1024 tokens, which affected 5 of the 100 prompts.
- Batch sizes were 10 for 0.6B and 2 for 4B, in fp16 on MPS. Loading 4B took 57 s, and the warm-up query took 82 s.
