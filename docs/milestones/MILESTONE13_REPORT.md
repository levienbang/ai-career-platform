# Milestone 13 — Taxonomy Docker, index chịu quota, học alias skill

Triển khai theo `AGENTS.md` và `docs/CURRENT_MILESTONE.md`, trên working tree M12
chưa commit. Không sửa `docs/PROJECT_SPEC.md` hoặc repo `../Crawl`, không thêm
runtime dependency, không ghi secret/CV/private data và không commit thay người dùng.

## Kết quả theo phase

- **Phase 0:** baseline 249 passed, 1 skipped; Ruff lint/format đều pass.
- **Phase 1:** `SKILL_DATA_DIR` mặc định `data`; giá trị rỗng dùng mặc định. File
  taxonomy thiếu báo đường dẫn và cách cấu hình. Seed/merge/learn/review dùng
  đường dẫn tính theo vị trí script, hoạt động cả khi cài non-editable và đổi cwd.
- **Phase 2:** embed/upsert ngay từng lô; retry chỉ lỗi embedding 429/5xx, đọc cả
  lỗi provider được wrap trong exception chain. Numeric `retry-after` ưu tiên hơn
  backoff mũ (cap 120 giây); RPM throttle/clock/sleep inject được. Hash SHA-256 của
  search text quyết định job cần embed; hỗ trợ `--full`, progress, skipped/deleted.
  Lỗi embedding nêu số job đã lưu; chạy lại tiếp tục từ lô chưa xong.
- **Phase 3:** migration `20261003_05`; metadata provenance cho alias và bảng
  `skill_decisions`. Gom tên grounded theo toàn request, giới hạn 5 từ/50 ký tự,
  gọi DeepSeek theo lô sau khi dựng curated + 10 ứng viên gần theo chuỗi.
  Kiểm tra schema, số item, ID ứng viên, confidence và blocklist trước khi ghi.
  Lỗi model không chặn import và không cache quyết định lỗi. Resolve đọc lại
  taxonomy sau learning để job trong cùng request dùng canonical mới.
  Có script học skill cũ và script list/approve/keep-new/revert.
- **Phase 4:** đổi tên hiển thị về canonical của alias file nếu không đụng unique;
  tên cũ thành alias. Các cặp blocklist bị chặn ở cả plan và apply merge.
- **Phase 5:** cập nhật README, biến môi trường, báo cáo này và Serena conventions
  với các quy ước bền vững của M13.

## File tạo/sửa trong lượt M13

| File | Thay đổi |
|---|---|
| `app/config.py`, `.env.example`, `docker-compose.yml` | Path, learning và embedding configuration |
| `app/db/models.py` | Alias provenance và model `SkillDecision` |
| `migrations/versions/20261003_05_skill_alias_learning.py` (mới) | Upgrade/downgrade schema |
| `app/services/skill_taxonomy.py` (file M12 có sẵn trong working tree) | Path rõ ràng, blocklist, display canonical, provenance, chuyển quyết định khi merge, giữ evidence |
| `app/ingestion/normalizer.py` | Gom skill lạ có evidence; xét lại extracted chưa có decision sau lỗi LLM |
| `app/ingestion/pipeline.py` | Batch learning trước persist; savepoint từng quyết định; log `alias_llm_calls` riêng |
| `app/services/skill_learning.py` (mới) | Schema/prompt/candidates/decision cache và áp learning |
| `app/services/skill_review.py` (mới) | Logic duyệt/gỡ ngoài CLI, không gọi model |
| `app/retrieval/qdrant.py` | Batches/retry/throttle/hash/incremental indexing |
| `app/api/search.py` | Truyền Settings của API cho index service |
| `data/skill_alias_blocklist.json` (mới) | 12 cặp cấm gộp, áp hai chiều theo match key |
| `scripts/seed.py`, `scripts/merge_skills.py` (M12) | Đường dẫn script-relative và alias provenance/blocklist |
| `scripts/index_jobs.py` | `--full`, progress và skipped/deleted |
| `scripts/learn_skill_aliases.py` (mới) | Dry-run/apply skill extracted cũ; đếm LLM calls |
| `scripts/review_skill_decisions.py` (mới) | CLI list/approve/keep-new/revert |
| `tests/conftest.py` | Chặn alias model thật; test phải inject fake model |
| `tests/test_dense_retrieval.py` | Kỳ vọng lần index thứ hai skipped thay vì embed lại |
| `tests/test_milestone13_index.py` (mới) | Path, 45-job batching, retry/partial progress, RPM, hash/full/deletion |
| `tests/test_milestone13_learning.py` (mới) | Import/cache/fallback/batching/candidates/scripts/review/blocklist/CV/provenance |
| `tests/test_milestone13_migration.py` (mới) | Upgrade/downgrade/defaults/constraints/FK |
| `README.md`, `docs/MILESTONE13_REPORT.md` (mới) | Workflow và giải thích vận hành |
| `.serena/memories/conventions.md` | Quy ước bền vững về learning, blocklist và incremental index |

Các thay đổi M12 có sẵn khác, gồm Dockerfile, extractor, structured parsing,
error handlers và test M12, được giữ lại. Không ghi đè `.Rhistory` hoặc dữ liệu
người dùng; không xem chúng là deliverable M13.

## Quyết định kỹ thuật

1. Learning chỉ chạy sau evidence/length checks, dùng model DeepSeek cấu hình
   chung và LLM throttle hiện có; không dùng Google embedding cho candidates.
   Prompt phân biệt synonym với part-of/variant/language family. Toàn bộ dữ liệu
   gửi model là untrusted; output tiếp tục được kiểm tra ở service.
2. Tự ghi alias/new khi confidence >= 0.9 mặc định. Low-confidence, invalid ID
   hoặc blocked pair -> extracted + pending. Alias/new/pending/rejected được cache
   bằng `skill_match_key`, tránh lặp API. Timeout/schema lỗi -> warning an toàn,
   không decision; import tiếp theo được thử lại, kể cả skill đã tạo extracted.
3. Script dry-run chỉ đọc DB nhưng vẫn có thể tốn lượt DeepSeek. `--apply` nằm
   trong một transaction. Import dùng savepoint cho từng quyết định và từng job.
4. Provenance: alias từ seed là curated; consolidation/đổi tên canonical theo
   file là merge; synonym được model hoặc reviewer approve là llm, có confidence
   và reason. Seed cập nhật provenance cho alias thuộc file curated.
5. Merge chuyển `job_skills`, aliases và decision FK, giải quyết unique collisions
   và giữ evidence bị trùng trước khi xoá link. Revert chỉ LLM alias; chỉ tách job
   có đúng wording trong evidence. Canonical claim riêng được giữ nếu còn evidence
   ngoài wording bị revert (tránh nhận nhầm `Tool` bên trong `Tool lib`).
6. Index dùng hash search text, giữ point ID theo job ID; partial success tồn tại
   ở Qdrant nên không cần queue hoặc thêm bảng checkpoint. Job không đổi -> 0 embed
   lần sau; `--full` dành cho thay model hoặc rebuild chủ động.

## Commands vận hành

```bash
docker compose up --build -d
docker compose exec api alembic upgrade head
docker compose exec api python scripts/seed.py --taxonomy-only
# Import JSON/CSV qua POST /jobs/import.
docker compose exec api python scripts/learn_skill_aliases.py
# Dry-run vẫn gọi DeepSeek. --apply mới ghi DB:
docker compose exec api python scripts/learn_skill_aliases.py --apply
docker compose exec api python scripts/merge_skills.py --apply
docker compose exec api python scripts/index_jobs.py
# Ép embed lại mọi job khi cần:
docker compose exec api python scripts/index_jobs.py --full
```

Review: `python scripts/review_skill_decisions.py list --decision pending`,
`approve "name" --as "canonical"`, `keep-new "name"`, `revert "alias"`.
Sau apply/review/revert, chạy index tăng dần trước khi so CV match.

## Kiểm tra và kết quả thực tế

| Kiểm tra | Kết quả |
|---|---|
| Baseline `UV_CACHE_DIR=/private/tmp/m13-uv uv run pytest` | 249 passed, 1 skipped, 1.36s |
| Baseline Ruff check / format --check | Pass; 129 files formatted |
| Focused phase 1/2 + regressions | 35 passed |
| Focused toàn bộ test M13 | 51 passed, 0.78s |
| Final host `UV_CACHE_DIR=/private/tmp/m13-uv uv run pytest` | **300 passed, 1 skipped**, 2.12s |
| Final `uv run ruff check .` (cùng cache override) | **All checks passed** |
| Final `uv run ruff format --check .` (cùng cache override) | **138 files already formatted** |
| `git diff --check` | Pass |
| Docker full suite, bật `RUN_DOCKER_E2E=1` | **301 passed**, 2.31s; không skip |
| Migration SQLite upgrade/defaults/constraints/FK/downgrade | Pass trong test M13 |
| Migration PostgreSQL upgrade -> downgrade -> upgrade | Pass, head `20261003_05` |
| Legacy alias PostgreSQL sau upgrade | `source=curated`; created_at được điền |
| Docker seed taxonomy-only | `Taxonomy is ready.`; exit 0 |
| Docker merge dry-run | `Dry-run (no database changes): 0 merge group(s)`; exit 0 |
| Docker seed/merge từ cwd `/tmp` | Cả hai exit 0 |
| Docker learning CLI từ cwd `/tmp`, không có extracted cần xét | 0 decisions, 0 LLM calls; exit 0 |

Test index dùng fake embedder và clock/sleep, Qdrant in-memory; learning dùng fake
model. Test `/cv/match` xác nhận CV Scikit-learn khớp job Scikit learn lib đã học.
Blocklist được test đủ 12 cặp ở cả hai chiều. Cũng có kiểm tra alias trỏ tới skill
vừa được merge trong cùng batch và revert không giữ canonical claim giả.

Test E2E mặc định trên host bị skip vì cần `RUN_DOCKER_E2E=1`. Đã bật và chạy nó
trong Docker với PostgreSQL/Qdrant thật, fake extractor/embedder/router/reranker;
luồng import -> index -> search -> agent grounded answer pass, không gọi API AI.

### Docker verification độc lập

Dùng Compose project riêng `m13-verification`, PostgreSQL database tạm trên tmpfs
và Qdrant riêng, không bind data/port của project hiện tại. Image xác minh lấy
`llm-api:latest` sẵn có làm dependency runtime, copy source hiện tại rồi chạy
`pip install --no-cache-dir --no-deps .` (non-editable). Xác nhận module từ cwd
`/tmp` ở `/usr/local/lib/python3.12/site-packages/app/services/skill_taxonomy.py`.

Các lệnh thực tế có prefix
`docker compose -p m13-verification -f /private/tmp/m13-compose.yml`:

```bash
exec -T api alembic upgrade 20261002_04
# Thêm một alias synthetic bằng SQL ở revision cũ.
exec -T api alembic upgrade head
# Assert source='curated' và created_at cho alias synthetic.
exec -T api alembic downgrade 20261002_04
exec -T api alembic upgrade head
exec -T api python scripts/seed.py --taxonomy-only
exec -T api python scripts/merge_skills.py
exec -T -w /tmp api python /app/scripts/seed.py --taxonomy-only
exec -T -w /tmp api python /app/scripts/merge_skills.py
exec -T -e RUN_DOCKER_E2E=1 api python -m pytest
exec -T -w /tmp api python /app/scripts/learn_skill_aliases.py
down --volumes --rmi local
```

DB `career_ai` và các container project `llm` hiện tại không được thay đổi. Không
chạy learning/index thật trên 99 job hoặc CV cá nhân trong lượt này.

### Lệnh ban đầu không hoàn tất và lỗi đã sửa

- `uv run pytest` và Ruff ban đầu không đọc được cache sandbox tại
  `~/.cache/uv`; đã chạy lại thành công với `UV_CACHE_DIR=/private/tmp/m13-uv`.
- Docker socket ban đầu bị sandbox chặn; đã dùng escalation được auto-review
  cho phép và thực hiện kiểm tra trên project tạm.
- Build từ `python:3.12-slim` mới bị dừng chủ động (exit 130) do tải lại bộ
  dependency lớn quá chậm. Docker checks sau đó dùng runtime dependency image
  có sẵn + source M13 cài non-editable. Chưa xác minh lại một clean build tải
  toàn bộ dependency mới từ PyPI; Dockerfile của M12 được giữ nguyên.
- Một assertion test mới ban đầu tính cả merge Elasticsearch không liên quan;
  đã thu hẹp assertion vào cặp Java/JavaScript bị chặn. Thêm log alias calls
  ban đầu làm test M10 kiểm tra chuỗi log fail; đã chuyển field mới về cuối,
  giữ định dạng cũ. Các lượt final toàn bộ suite pass như bảng trên.

## Giới hạn còn lại và khái niệm cần hiểu

- Không đo quota/latency/chất lượng DeepSeek/Google thật hoặc CV riêng. Các bước
  trên DB 99 job, đếm API calls và so CV trước/sau để Claude chạy tiếp như scope.
- RPM spacing không đảm bảo token/phút hoặc ngày. Hết retries thì dừng nhưng
  giữ lô đã xong; chỉnh batch/RPM theo quota thật. Retry-after hiện đọc giá trị
  số giây; header HTTP-date không được dùng và sẽ rơi về backoff.
- LLM confidence không phải xác suất đã hiệu chuẩn: blocklist và review vẫn cần.
  Pending/rejected được nhớ lâu dài; script approve/keep-new là cách thay đổi.
- Revert dựa vào evidence hiện có, không khôi phục nguyên trạng các requirement
  trước khi merge hoặc các evidence vốn đã thiếu. Preferred/required collision
  vẫn ưu tiên required như M12. Không có UI hoặc history table trong scope.
- Hash chỉ phản ánh search text theo yêu cầu. Thay embedding model cùng dimension
  cần `--full`; khác dimension cần rebuild collection. Metadata ngoài search text
  không tự buộc embed lại. PostgreSQL/Qdrant chỉ đồng bộ khi chạy index.
- Downgrade xoá toàn bộ decisions và alias provenance, mất dữ liệu học được.
- Cần hiểu khác biệt canonical skill/alias/decision, synonym và quan hệ part-of,
  evidence grounding, transaction/savepoint, và incremental hash indexing.
