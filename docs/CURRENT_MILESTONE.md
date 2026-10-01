# Current milestone — Milestone 12: Chất lượng skill, log lỗi và build nhanh

> File này là yêu cầu được user cho phép cho milestone hiện tại (ghi đè
> Milestone 11 đã hoàn thành; báo cáo M11 ở `docs/MILESTONE11_REPORT.md`). Với
> milestone này, nó override các điểm tương ứng trong `docs/PROJECT_SPEC.md` mục
> 8.2 (skill normalization). **Không sửa repo Crawl (`../Crawl`)** và không sửa
> `docs/PROJECT_SPEC.md`.
>
> Code M9–M11 có thể chưa được commit. Xây tiếp trên code hiện tại, giữ nguyên
> hành vi M9–M11 trừ những điểm file này nói khác. Chat chỉ dùng DeepSeek
> (`DEEPSEEK_MODEL` duy nhất); embedding chỉ dùng Google. **Không ghi API key
> vào bất kỳ file nào được commit.**

## 1. Bối cảnh: kết quả chạy thật

Đã chạy thật Crawl → `/jobs/import` (DeepSeek) → Postgres → Qdrant →
`/cv/match` với 100 job VietJobs AI/ML/Data và CV của user. Kết quả xếp hạng hợp
lý, nhưng có các vấn đề sau:

1. **Skill trùng vì khác cách viết** — trong DB thật có các cặp là hai skill
   riêng: `Node.js`/`Nodejs`, `Power BI`/`PowerBI`, `Hugging Face`/`HuggingFace`,
   `Elastic Search`/`ElasticSearch`, `PL/SQL`/`PLSQL`, `Data Stage`/`Datastage`.
2. **Skill trùng vì đồng nghĩa** — `sklearn` và `Scikit-learn` là hai skill:
   `/cv/match` báo user **thiếu `sklearn`** dù CV có `Scikit-learn` → `skill_score`
   thấp oan. Seed taxonomy (`data/seed_data.json`) chỉ có alias cho 6 skill.
3. **"Skill" là cả câu** — từ danh sách skill nguồn (đường deterministic M10),
   ví dụ `Sử dụng thành thạo các công cụ BI như Power BI, Metabase, Google Data Studio`,
   `Thư viện học máy (Scikit-learn, TensorFlow, Keras, PyTorch)`.
4. **Lỗi extraction không có log** — `LangChainJobExtractor.extract_many`/`extract`
   nuốt exception và chỉ trả `"Structured extraction failed"`. Lần chạy thật bị
   reject 50 job do `.env` cũ gửi tên model Gemini sang DeepSeek (HTTP 400),
   nhưng không có dòng log nào cho biết nguyên nhân.
5. **`/cv/match` trả 200 + danh sách rỗng khi Qdrant collection tồn tại nhưng
   chưa có điểm nào** (index thất bại giữa chừng) → người dùng tưởng không có job
   phù hợp.
6. **Rebuild Docker > 10 phút** — `Dockerfile` copy `app/` trước
   `pip install`, nên mỗi lần sửa code là cài lại toàn bộ dependency (Docling,
   torch…).

## 2. Phase 0 — Baseline

- Chạy `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` và
  ghi kết quả trước khi sửa.
- Test không gọi mạng hay API trả phí.

## 3. Phase 1 — Khoá so khớp skill không phụ thuộc cách viết

### 3.1 `skill_match_key` (`app/ingestion/normalizer.py`)

Thêm hàm `skill_match_key(value: str) -> str`:

- Bắt đầu từ `skill_lookup_key` hiện có (NFKC, gộp khoảng trắng, casefold).
- Bỏ khoảng trắng, `.`, `-`, `_`, `/`.
- **Giữ** `+` và `#` (để `C`, `C++`, `C#` khác nhau).
- Ví dụ: `Node.js`, `NodeJS`, `node js` → `nodejs`; `Power BI`, `PowerBI` →
  `powerbi`; `PL/SQL`, `PLSQL` → `plsql`; `C++` → `c++`; `C#` → `c#`; `.NET` →
  `net`; `Scikit-learn` → `scikitlearn`.

### 3.2 Dùng khoá mới khi resolve

- `SkillNormalizer` dựng taxonomy theo `skill_match_key` cho canonical name và
  mọi alias; `resolve()` và `normalize()` tra theo `skill_match_key`.
- **Kiểm tra bằng chứng (M9) giữ nguyên dùng `skill_lookup_key` + biên từ** trên
  text gốc — không nới lỏng chống bịa.
- Khi tạo skill `extracted` mới mà match key đã tồn tại → dùng skill có sẵn,
  không tạo bản mới.
- CV (`recognize_cv_skills`) dùng cùng resolver nên tự hưởng lợi.

## 4. Phase 2 — Alias đồng nghĩa có kiểm soát

### 4.1 Dữ liệu alias

- File mới `data/skill_aliases.json`:
  `{"<Canonical>": {"category": "<cat|null>", "aliases": ["...", ...]}}`.
- Khoảng 40–60 nhóm phổ biến trong job IT/AI/Data, tối thiểu gồm:
  `Scikit-learn` (sklearn, scikit learn), `TensorFlow` (tf, tensorflow 2),
  `PyTorch` (torch), `Keras`, `Kubernetes` (k8s), `PostgreSQL` (postgres),
  `JavaScript` (js), `TypeScript` (ts), `Node.js`, `React` (reactjs, react.js),
  `Next.js`, `Vue.js` (vuejs, vue), `Microsoft SQL Server` (sql server, mssql,
  ms sql), `Power BI`, `Adobe Photoshop` (photoshop), `Adobe Illustrator`
  (illustrator), `Go` (golang), `Amazon Web Services` (aws), `Google Cloud
  Platform` (gcp), `Microsoft Azure` (azure), `CI/CD`, `Machine Learning`
  (ml), `Deep Learning` (dl), `Natural Language Processing` (nlp), `Large
  Language Models` (llm, llms), `Computer Vision`, `Hugging Face`, `Apache
  Spark` (spark, pyspark), `Apache Airflow` (airflow), `Apache Kafka` (kafka),
  `Elasticsearch` (elastic search), `MongoDB` (mongo), `Git`, `GitHub`,
  `Docker`, `Linux`, `Pandas`, `NumPy`, `XGBoost`, `LightGBM`.
- **Không** thêm alias mơ hồ có thể đụng nghĩa khác (ví dụ `r`, `c`, `go` đứng
  một mình phải là canonical riêng, không làm alias của skill khác).
- Hai alias không được trỏ tới hai canonical khác nhau theo `skill_match_key`
  (test kiểm tra).

### 4.2 Seed

- `scripts/seed.py` (kể cả `--taxonomy-only`) nạp thêm `data/skill_aliases.json`
  sau taxonomy hiện có; idempotent; skill được seed là `origin="curated"`.
- Nếu một skill `extracted` đã tồn tại khớp canonical hoặc alias theo
  `skill_match_key` → gộp vào canonical (xem 4.3), không tạo bản trùng.

### 4.3 Gộp skill trùng đang có trong DB

Script mới `scripts/merge_skills.py`:

- Nhóm skill theo: cùng `skill_match_key`, hoặc khớp cùng một canonical/alias
  trong `data/skill_aliases.json`.
- Chọn skill giữ lại: canonical trong alias file → nếu không có thì `curated` →
  nếu không có thì skill có nhiều `job_skills` nhất → tie-break `id` nhỏ nhất.
- Chuyển `job_skills` sang skill giữ lại; nếu đụng unique
  `(job_id, skill_id, requirement_type)` thì giữ một dòng (ưu tiên `required`
  hơn `preferred` khi cùng job — xoá dòng `preferred` thừa).
- Tên của skill bị gộp được thêm thành `SkillAlias` của skill giữ lại; xoá skill
  bị gộp.
- Mặc định **dry-run** (in ra các nhóm sẽ gộp và số `job_skills` bị ảnh
  hưởng); chỉ ghi DB với `--apply`; chạy trong một transaction; chạy lại không
  đổi gì.
- In nhắc: sau `--apply` cần chạy lại `scripts/index_jobs.py` vì search
  document chứa tên skill.

## 5. Phase 3 — Không tạo skill dạng câu

Áp dụng **chỉ khi tạo skill `extracted` mới** (skill curated/alias không đổi):

- Bỏ tên skill có **hơn 5 từ** (đếm theo khoảng trắng sau khi làm sạch) hoặc
  dài hơn **50 ký tự**.
- Với đường deterministic (`app/ingestion/structured.py::parse_skill_list`):
  phần tử dạng `"<cụm chữ> (<A>, <B>, ...)"` mà phần trong ngoặc là danh sách
  ngăn bởi dấu phẩy → **thay bằng các phần tử trong ngoặc** (ví dụ
  `Thư viện học máy (Scikit-learn, TensorFlow, Keras, PyTorch)` → 4 skill). Phần
  tử sau đó vẫn qua luật 5 từ / 50 ký tự và kiểm tra bằng chứng.
- Skill bị bỏ không làm job bị reject (giữ hành vi M9).

## 6. Phase 4 — Log lỗi extraction

- `app/ingestion/extractor.py`: mọi chỗ bắt exception khi gọi model
  (`extract`, `extract_many`, kể cả khi chia đôi batch) ghi
  `LOGGER.warning(...)` gồm: tên component, số record trong lời gọi, lần thử,
  **tên lớp exception** và **thông điệp lỗi đã cắt ngắn (≤ 300 ký tự)**.
- **Không log** nội dung job, prompt, output model hay API key. Nếu thông điệp
  lỗi có chuỗi giống key (`sk-…`), thay bằng `***`.
- Khi một record cuối cùng thành `None`, log thêm một dòng tổng kết
  (`n records failed after retries/split`). `ImportResult` giữ nguyên shape.
- Logger cấu hình để dòng `WARNING` hiện trong `docker compose logs api`.

## 7. Phase 5 — `/cv/match` khi index rỗng

- `DenseSearchService` (hoặc `CVMatchService`): collection tồn tại nhưng
  `points_count == 0` → raise `SearchIndexNotReadyError` với thông báo
  `"Qdrant collection '<name>' is empty; index jobs first"` → API trả **503**
  như trường hợp chưa có collection.
- Áp dụng cho các endpoint search dùng dense search (cùng một kiểm tra).

## 8. Phase 6 — Dockerfile build nhanh

- Tách layer: cài dependency **trước** khi copy `app/`, `migrations/`,
  `scripts/`, `data/`, `evaluation/`. Có thể dùng `uv.lock` (`uv sync --frozen
  --no-install-project` rồi copy code và cài project) hoặc cách tương đương; giữ
  `ARG INSTALL_TARGET` để `docker-compose.dev.yml` cài `.[dev]` vẫn chạy.
- Kết quả chạy giữ nguyên: `alembic upgrade head` + `uvicorn` như hiện tại.
- Kiểm tra: build lần 2 sau khi chỉ sửa một file trong `app/` phải dùng cache cho
  layer dependency (ghi thời gian build trước/sau trong báo cáo).

## 9. Ngoài phạm vi

- Index chia lô / retry 429 / embed tăng dần (user dùng key Google có quota đủ).
- Đổi provider, đổi công thức `/cv/match`, đổi schema `jobs`.
- Sửa repo Crawl.

## 10. Tests bắt buộc (không gọi mạng)

1. `skill_match_key`: các ví dụ ở 3.1; `C`, `C++`, `C#` khác nhau.
2. Resolver: `Nodejs`/`node js` resolve về `Node.js`; tạo skill extracted
   `PowerBI` khi đã có `Power BI` → dùng skill có sẵn.
3. Bằng chứng: skill khớp match key nhưng không xuất hiện trong text job → vẫn
   bị bỏ (chống bịa không đổi).
4. Alias file: JSON hợp lệ; không có hai canonical trùng match key; không alias
   nào trỏ tới hai canonical.
5. Seed idempotent; `--taxonomy-only` nạp alias; skill extracted trùng được gộp.
6. `merge_skills.py`: dry-run không ghi DB; `--apply` gộp
   `sklearn`→`Scikit-learn`, `Nodejs`→`Node.js`, chuyển `job_skills`, xử lý đụng
   unique (required thắng preferred), thêm alias, chạy lại không đổi.
7. CV: CV có `Scikit-learn`, job yêu cầu `sklearn` (sau khi gộp hoặc qua alias)
   → `matched_skills` chứa skill đó, không nằm trong missing.
8. Skill dạng câu: > 5 từ hoặc > 50 ký tự không được tạo; phần tử có ngoặc liệt
   kê được tách đúng; job vẫn insert.
9. Log: extractor với fake model raise lỗi → có dòng `WARNING` chứa tên
   exception, không chứa nội dung job; chuỗi `sk-…` bị che.
10. `/cv/match` với collection rỗng → 503 "is empty; index jobs first".
11. Toàn bộ test M9–M11 pass không đổi.

## 11. Definition of done

- Toàn bộ test, `ruff check`, `ruff format --check` pass.
- Image Docker build được; báo cáo thời gian rebuild sau khi sửa một file `app/`.
- Báo cáo theo mục "Final response" của `AGENTS.md`, lưu thêm
  `docs/MILESTONE12_REPORT.md`.
- Sau đó Claude sẽ: chạy `merge_skills.py` (dry-run rồi `--apply`) trên DB
  `career_ai`, re-index, chạy lại `/cv/match` với CV của user để so sánh
  `skill_score` trước/sau (đặc biệt cặp `Scikit-learn`/`sklearn`).
