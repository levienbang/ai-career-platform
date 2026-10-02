# Milestone 10 — Báo cáo triển khai

Đã thực hiện Phase 0–5 trên code M9 hiện có, giữ nguyên các thay đổi chưa commit.
Không sửa `../Crawl` hoặc `docs/PROJECT_SPEC.md`; không gọi LLM/embedding trả phí.

## File tạo mới

- `app/ingestion/structured.py`: parser skill/experience và deterministic extraction.
- `migrations/versions/20261002_04_job_raw_hash.py`: raw hash nullable + unique constraint.
- `tests/test_milestone10_ingestion.py`: raw hash, migration, parser, dedup, routing,
  savepoint, log, cấu hình và API batch.
- `tests/test_milestone10_extractor.py`: prompt, guard, index, retry/split và throttle.
- `tests/test_milestone10_cv_seed.py`: query CV và taxonomy-only seed.
- `docs/MILESTONE10_REPORT.md`: báo cáo này.

## File sửa trong lượt này

- `app/config.py`, `.env.example`, `docker-compose.yml`: ba biến cấu hình import mới.
- `app/db/models.py`, `app/db/repositories.py`: `Job.raw_hash` và lookup.
- `app/ingestion/cleaner.py`: giữ newline phân cách skill nguồn.
- `app/ingestion/pipeline.py`: prepare → extract → persist; index gốc và log route.
- `app/ingestion/extractor.py`: single/batch extraction, guard, retry/split, throttle.
- `app/llm.py`: tùy chọn `include_raw` để cứu item hợp lệ khi parser của provider
  từ chối index thiếu/sai của item khác; các component khác giữ mặc định cũ.
- `app/schemas/ingestion.py`: schema batch, `extra="forbid"`.
- `app/api/jobs.py`: truyền settings vào pipeline trong threadpool hiện có của M9.
- `app/services/cv_match.py`: bổ sung experience/education vào query 2.000 ký tự.
- `scripts/seed.py`: `--taxonomy-only`, mặc định vẫn seed demo jobs.
- `tests/conftest.py`, `tests/test_ingestion_pipeline.py`: fake batch và lỗi extraction
  theo hợp đồng mới (`None` → `Structured extraction failed`).
- `README.md`: luồng import, cấu hình, request budget, Crawl và seed dữ liệu thật.
- `docs/CURRENT_MILESTONE.md`: chỉ thêm dòng trống trong code block để format pass.
- `.serena/memories/conventions.md`: cập nhật kiến thức kiến trúc bền vững của M10.

## Quyết định và luồng dữ liệu

1. Một transaction cho request; prepare không ghi DB hoặc gọi LLM. URL/raw hash
   loại trùng trước extraction, kể cả trùng trong request. Content hash vẫn kiểm
   tra business duplicate trước insert; mỗi record lưu có savepoint riêng.
2. Nguồn có title, description và skill parse được dùng deterministic extraction.
   Skill `technical_skills` không phân loại được lưu là required. Evidence dùng
   title, description và technical-skills gốc; không tự hợp nhất alias mới.
3. Record còn lại gom batch tuần tự. Kết quả ánh xạ theo index, index thiếu/trùng/
   ngoài phạm vi không được gán nhầm sang record khác. Batch lỗi toàn bộ retry,
   sau đó chia đôi; singleton lỗi trả `None`. Errors giữ index gốc và được sort.
4. Source company luôn thắng; các title/location/employment type/source URL có
   giá trị trong input thắng model. Prompt chỉ gửi sáu field cơ bản + technical
   skills. Không gửi salary/benefits/category và không log nội dung job.
5. RPM limiter dùng monotonic clock/sleep có thể inject, chia sẻ trong cùng
   process và áp dụng cả retry/split. Không thêm parallel LLM, queue hoặc dependency.
6. Query CV dùng summary → skill names → projects → experience → education;
   loại raw CV và skill evidence. Công thức scoring M9 giữ nguyên.

## Kết quả kiểm tra thực tế

| Kiểm tra | Kết quả |
| --- | --- |
| Phase 0 baseline pytest | 128 passed, 1 skipped |
| Phase 0 baseline ruff check | Pass |
| Phase 0 baseline format | Fail: thiếu một dòng trống trong code block milestone |
| Phase 1 focused tests | 9 passed |
| Phase 2 focused tests | 32 passed |
| Bộ test M10 cuối cùng | 46 passed |
| Full pytest cuối cùng | 174 passed, 1 skipped, 0.92s |
| E2E bật dịch vụ thật, chạy riêng | 1 passed, 0.26s |
| Ruff check cuối cùng | Pass |
| Ruff format cuối cùng | Pass: 121 files already formatted |
| PostgreSQL migration | Upgrade, NULL legacy hash, unique constraint, downgrade pass |
| Docker Compose config | Pass |
| git diff --check | Pass |

Các lệnh kiểm tra dùng môi trường đã cài:

```bash
UV_CACHE_DIR=/private/tmp/llm-uv-cache uv run --no-sync pytest -rs
UV_CACHE_DIR=/private/tmp/llm-uv-cache uv run --no-sync ruff check .
UV_CACHE_DIR=/private/tmp/llm-uv-cache uv run --no-sync ruff format --check .
UV_CACHE_DIR=/private/tmp/llm-uv-cache uv run --no-sync pytest \
  tests/test_milestone10_ingestion.py tests/test_milestone10_extractor.py \
  tests/test_milestone10_cv_seed.py
docker compose config --quiet
git diff --check
```

`uv run` nguyên bản không chạy được vì sandbox không ghi được cache mặc định;
đổi cache sang `/private/tmp` thì bước sync cần mạng bị hạn chế. Vì vậy dùng
`--no-sync` với `.venv` đã cài. Docker socket được truy cập sau khi escalation
được chấp thuận. URL DB local ban đầu dùng hostname Docker `postgres` nên không
resolve từ host; kiểm tra PostgreSQL chuyển riêng host sang `127.0.0.1`.

Full suite mặc định skip `tests/test_e2e_pipeline.py` vì thiếu cờ
`RUN_DOCKER_E2E=1`. Đã bật cờ và chạy riêng test đó với PostgreSQL schema UUID
mới và Qdrant collection riêng, dùng fake extractor/embedding/router/reranker.
Migration chạy riêng bằng Alembic Operations trong PostgreSQL schema tạm có
transaction rollback; schema E2E được dọn sau test. Không sửa dữ liệu hiện có.
Runner tạm: `/private/tmp/llm_m10_pg_verify.py` và
`/private/tmp/llm_m10_e2e_verify.py`, chạy bằng `uv run --no-sync python` với
`PYTHONPATH` trỏ repository và quyền kết nối dịch vụ local.

## Chạy bản cập nhật

Đối với Docker, rebuild API để lấy code/cấu hình mới và migration startup:

```bash
docker compose up --build -d
docker compose exec api alembic upgrade head
# Chỉ taxonomy trên DB mới khi đánh giá dữ liệu thật:
docker compose exec api python -m scripts.seed --taxonomy-only
# Sau khi Crawl deliver theo hướng dẫn README:
docker compose exec api python -m scripts.index_jobs
curl -F 'file=@cv.pdf' -F 'limit=10' http://localhost:8000/cv/match
```

Không chạy seed trên DB người dùng trong lượt này; không restart API đang chạy
hoặc áp migration lên schema dữ liệu hiện có. README có lệnh local tương ứng;
`DATABASE_URL` dùng `localhost` khi chạy trên host, `postgres` trong Docker.

## Giới hạn và khái niệm cần hiểu

- Chưa import bộ ~2.040 VietJobs hoặc đo request Gemini/Gemma thật. Khả năng
  tiết kiệm request được xác minh bằng fake; cần đối chiếu log route trên dữ liệu thật.
- Job cũ có raw hash NULL, nên không dedup trước LLM bằng raw hash cho đến khi
  có record mới tương ứng; URL/content hash vẫn bảo vệ dữ liệu như trước.
- Ước lượng `ceil(llm_records / batch_size) × (1 + retries)` không bao gồm các
  batch con khi split fallback. Deterministic records và raw duplicates tốn 0 call.
- RPM là giới hạn theo process; nhiều worker/deployment cần chia quota phù hợp.
  Request/ngày và token/phút không được tự quản lý trong milestone này.
- Seed taxonomy-only không xóa demo jobs đã có. Chỉ dùng DB mới nếu cần tập dữ
  liệu đánh giá sạch. PostgreSQL lưu dữ liệu, Qdrant cần indexing riêng sau import.
- Raw hash đo giống nhau ở input; content hash đo giống nhau sau chuẩn hóa nghiệp
  vụ. Batch giảm số request, throttle kiểm soát nhịp, deterministic tránh inference
  khi nguồn đã có cấu trúc. Savepoint cô lập lỗi lưu của một record.
