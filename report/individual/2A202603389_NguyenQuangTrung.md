# Báo cáo Day 7: Agentic RAG, Retrieval & Context Engineering

- **Sinh viên:** Nguyễn Quang Trung — **MSSV:** `2A202603389`
- **Mức thực hiện:** Phần cơ bản (85 điểm) + **Thử thách nâng cao 100+ trang (+15 điểm)**

---

## 1. Dữ liệu và benchmark

### Nguồn, quyền sử dụng, số trang và số chunk
- Kế thừa trực tiếp bộ dữ liệu và chiến lược chunking **Strategy B** từ bài lab Day 6 (`D:\Vin\K5-L3A-Day06-NguyenQuangTrung-2A202603389-DataIngestion\results\chunks_strategy_B.json`), xuất chuẩn hóa sang [`starter/data/converted/chunks.json`](../../starter/data/converted/chunks.json) bằng [`src/prepare_day6_data.py`](../../src/prepare_day6_data.py):
  - `nist-ai-rmf-1.0` (`starter/data/NIST.AI.100-1.pdf` — *NIST AI Risk Management Framework 1.0*, 2023, tài liệu công khai của NIST / Chính phủ Hoa Kỳ): **40 trang** có nội dung được index (trang 6–47), **153 chunks**.
  - `nist-csf-2.0` (`starter/data/NIST.CSWP.29.pdf` — *NIST Cybersecurity Framework 2.0*, 2024, tài liệu công khai của NIST / Chính phủ Hoa Kỳ): **27 trang** có nội dung được index (trang 5–31), **94 chunks**.
- **Tổng cộng:** **67 trang** có nội dung được index, **247 chunks**.

### Vector store, embedding model, phiên bản và số chiều
- **Vector store:** Tái sử dụng PostgreSQL + `pgvector` của Day 6 (Docker container `day6_student_starter-postgres-1`, kết nối qua `DATABASE_URL` tại `127.0.0.1:55433/day6_lab`, bảng `document_chunks`, lọc `strategy = 'B'`, độ đo cosine distance `embedding <=> %s::vector` trong [`src/retrievers.py`](../../src/retrievers.py)).
- **Embedding model:** `text-embedding-3-small` của OpenAI, **1536 chiều**, gọi qua OpenAI REST API (`https://api.openai.com/v1/embeddings`) trong [`src/embeddings.py`](../../src/embeddings.py) (đơn giá `$0.02 / 1M tokens`).

### Agent model, cách chạy và phân chia dev/test
- **Agent model:** **`gpt-6-luna`** qua OpenAI Chat Completions REST API (`https://api.openai.com/v1/chat/completions`), cấu hình `temperature = 0.0` và `reasoning_effort = "none"` khi truyền `tools` (yêu cầu bắt buộc của endpoint đối với `gpt-6-luna` khi dùng function calling), đơn giá `$0.10 / 1M input tokens` và `$0.50 / 1M output tokens` ([`src/agent.py`](../../src/agent.py), [`src/config.py`](../../src/config.py)).
- **Phân chia benchmark** ([`starter/data/converted/agent_cases.json`](../../starter/data/converted/agent_cases.json), tổng **24 cases**):
  - **Tập `dev` (17 cases):** 15 câu có đáp án (`q01`–`q15`) + 2 câu không đủ bằng chứng (`u_dev_01` về EU AI Act; `u_dev_02` về mapping sang ISO/IEC 42001:2023).
  - **Tập `test` holdout (7 cases):** 5 câu có đáp án (`q16`–`q20`) + 2 câu không đủ bằng chứng (`u_test_01` về biểu phí chứng nhận NIST CSF 2.0; `u_test_02` về mức phạt GDPR Article 22 trong NIST AI RMF 1.0).

### Cách xác định câu có/thiếu evidence và tiêu chí chấm
- **Câu có đáp án (`q01`–`q20`):** Kế thừa từ Day 6 kèm `gold` (`source_id`, `page`, `evidence`) và `alternative_gold`. Một chunk được tính là `hit` khi khớp `source_id`, `page`, và đoạn `evidence` nằm nguyên văn trong `chunk["text"]`.
- **Câu thiếu bằng chứng (`u_dev_01..02`, `u_test_01..02`):** Là các câu gần chủ đề quản trị AI / an ninh mạng nhưng chắc chắn vắng mặt trong bộ tài liệu gốc; **không** đưa vào mẫu số khi tính retrieval metrics.
- **Kỷ luật benchmark:** Không truyền `gold` vào agent hoặc query; thử nghiệm và chốt cấu hình trên `dev` trước khi chạy holdout `test`, không chỉnh sửa prompt/code sau khi đã quan sát kết quả `test`.
- **Tiêu chí audit thủ công trên `test`** (theo [`BENCHMARK.md`](../../BENCHMARK.md) và [`starter/agent_metrics.py`](../../starter/agent_metrics.py)):
  - **Context:** `0` = không có evidence; `1` = có nhưng thiếu điều kiện/nguồn; `2` = đủ evidence và nguồn.
  - **Grounded answer:** `0` = sai, thiếu evidence hoặc từ chối câu có đáp án; `1` = đúng một phần; `2` = đúng và đủ.
  - **Citation hợp lệ:** Citation nằm trong `final_context_ids` **và** nội dung chunk thực sự hỗ trợ các khẳng định đi kèm.
  - **Abstention:** Từ chối bằng tool `abstain` trên câu thiếu bằng chứng (trả lời bằng `submit_answer` với `status="answered"` bị chấm `abstention_correct = false`).

---

## 2. Retrieval và thử nghiệm

### 2.1. So sánh 4 phương pháp retrieval độc lập (`src/benchmark_retrieval.py`)
Chạy cùng corpus **247 chunks** (Day 6 Strategy B), `top_k = 3`, `context_budget = 2500` ký tự (kèm theo dõi `context_hit` ở ngưỡng `800` ký tự để đối chiếu với baseline):

| Phương pháp | Filter / tham số | Split | Hit@1 | Evidence Hit@3 | Context Hit (`budget=2500`) | Context Hit (`800`) | Median / Total (ms) | Chi phí (`$`) |
|---|---|---|---|---|---|---|---|---|
| `tfidf_no_filter` | Không dùng metadata filter, `top_k=3` | dev | `12/15` (80.0%) | `15/15` (100.0%) | `13/15` (86.7%) | `9/15` (60.0%) | 1.05 / 15.77 ms | `$0.00` |
| `bm25_filtered` | Okapi BM25 + filter từ `case["filter"]`, `top_k=3` | dev | `12/15` (80.0%) | `14/15` (93.3%) | `13/15` (86.7%) | `9/15` (60.0%) | 1.26 / 16.27 ms | `$0.00` |
| `dense_pgvector` | `text-embedding-3-small` (1536d) trên pgvector Day 6 + filter, `top_k=3` | dev | `12/15` (80.0%) | `13/15` (86.7%) | `12/15` (80.0%) | `9/15` (60.0%) | 705.97 / 16374.11 ms | `$7.98e-06` |
| `hybrid_rrf_dense_bm25` | RRF kết hợp `dense_pgvector` + `bm25_filtered`, `top_k=3` | dev | `12/15` (80.0%) | `14/15` (93.3%) | `13/15` (86.7%) | `9/15` (60.0%) | 713.72 / 10592.25 ms | `$7.98e-06` |
| `tfidf_no_filter` | Không filter, `top_k=3` | test | `2/5` (40.0%) | `3/5` (60.0%) | `3/5` (60.0%) | `0/5` (0.0%) | 1.15 / 6.51 ms | `$0.00` |
| `bm25_filtered` | BM25 + filter, `top_k=3` | test | `3/5` (60.0%) | `4/5` (80.0%) | `4/5` (80.0%) | `0/5` (0.0%) | 1.14 / 5.19 ms | `$0.00` |
| `dense_pgvector` | pgvector Day 6 + filter, `top_k=3` | test | `1/5` (20.0%) | `4/5` (80.0%) | `4/5` (80.0%) | `1/5` (20.0%) | 810.20 / 4370.34 ms | `$2.94e-06` |
| `hybrid_rrf_dense_bm25` | RRF Dense + BM25 + filter, `top_k=3` | test | `3/5` (60.0%) | **`5/5` (100.0%)** | `4/5` (80.0%) | `1/5` (20.0%) | 838.51 / 4268.57 ms | `$2.94e-06` |

- **Nhận xét retrieval:** Trên `test`, `hybrid_rrf_dense_bm25` phát huy rõ lợi thế khi kết hợp độ chính xác từ khóa (`GOVERN 1.6`, `RC.RP-03`) của BM25 với khả năng hiểu ngữ nghĩa của Dense, nâng `Evidence Hit@3` lên **`5/5` (100.0%)** so với `4/5` của từng phương pháp đơn lẻ và `3/5` của TF-IDF không filter. Chi phí embedding cho mỗi query rất nhỏ (`399` tokens trên dev, `147` tokens trên test, dưới `$0.00001`).

### 2.2. Thử nghiệm cấu hình Agent trên `dev` và đối chiếu `test` (`src/run_dev_experiments.py`, `src/run_test_evaluation.py`)

| Cấu hình | Phương pháp, top-k, filter, rerank/chọn context, budget | Giả thuyết / thay đổi | Split | Context Evidence Rate | Answered / Unans abstained | Avg search calls | Median latency / tổng chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (`cfg0_baseline_dense_800`) | `dense` mặc định, `max_searches=2`, `top_k=3`, không cho agent tự filter, `first_come`, `budget=800` ký tự | Mốc đối chiếu mặc định như starter | dev (17) | `11/15` (73.3%) | `9/15` / `2/2` | `1.65` | `5852.72 ms` / `$0.005733` |
| Cải thiện 1 (`cfg1_hybrid_2500_ranked`) | `hybrid` mặc định, `max_searches=3`, `top_k=3`, cho phép filter (`domain`/`year`), `score_ranked`, `budget=2500` | Đồng thời thử phương pháp Hybrid RRF + nới budget lên `2500` + xếp theo score | dev (17) | `14/15` (93.3%) | `15/15` / `2/2` | `1.41` | `4672.84 ms` / `$0.006809` |
| Cải thiện 2 (`cfg2_hybrid_4000_k5`) | `hybrid` mặc định, `max_searches=4`, `top_k=5`, cho phép filter, `score_ranked`, `budget=4000` | Tăng `top_k=5`, `max_searches=4`, `budget=4000` để tăng recall các câu nhiều ý | dev (17) | `15/15` (100.0%) | `14/15` / `2/2` | `1.59` | `4676.86 ms` / `$0.009577` |
| Cải thiện 3 — **Đã chọn** (`cfg4_hybrid_4000_k5_followup`) | Như `cfg2`, giữ `citation_retries<=1` + **follow-up search prompt guidance** (tách ý còn thiếu, không vội abstain sau 1 lượt) | Sửa tình trạng agent bỏ cuộc sớm hoặc thiếu mảnh bằng chứng thứ 2 | dev (17) | **`15/15` (100.0%)** | **`15/15`** / **`2/2`** | `1.59` | `4646.64 ms` / `$0.00895` |
| Cấu hình đã chọn (`cfg4_hybrid_4000_k5_followup`) | Giữ nguyên tham số đã chốt từ `dev` | Đánh giá holdout độc lập | test (7) | `3/5` (tự động) / audit: `0:0, 1:2, 2:3` | `5/5` / `1/2` | `2.29` | `9070.90 ms` / `$0.0059` |
| Baseline (`cfg0_baseline_dense_800`) | Giữ nguyên cấu hình baseline | Đối chiếu baseline trên holdout `test` | test (7) | `1/5` (tự động) / audit: `0:4, 1:1, 2:0` | `0/5` / `2/2` | `2.00` | `6350.26 ms` / `$0.002561` |

*(Ghi chú: `cfg1` vừa là thử nghiệm Hybrid RRF kết hợp nâng `budget=2500` và `score_ranked` nên đóng vai trò đồng thời phương pháp mới và cải thiện 1; `cfg2` và `cfg4` là hai cải thiện tiếp theo tách riêng tác động của mở rộng recall/budget và cải thiện chiến lược search/self-correction.)*

### 2.3. Hai lỗi dev đã phân tích và lý do chọn cấu hình
1. **Lỗi dev #1 — `q06` trong `cfg0_baseline_dense_800` (mất sạch context dù retriever tìm trúng rank 1):**
   - *Hiện tượng & trace:* Ở lượt 1 (`dense`) và lượt 2 (`bm25`), retriever đều xếp `nist-ai-rmf-1.0:p23:para:001` ở **rank 1**, nhưng `context_ids = []` và `context_chars = 0`, khiến agent buộc phải `abstain`.
   - *Nguyên nhân:* Độ dài riêng phần text của `nist-ai-rmf-1.0:p23:para:001` là **849 ký tự** (cộng header vượt xa `context_budget = 800`), nên `pack_context` trong [`src/retrievers.py`](../../src/retrievers.py) loại bỏ hoàn toàn mọi đoạn dài hơn 800 ký tự.
   - *Khắc phục & kết quả:* Tăng `context_budget` lên `2500` (`cfg1`) và `4000` (`cfg2`/`cfg4`) kết hợp `score_ranked`; `q06` có đủ context ngay ở lượt tìm đầu tiên và trả lời đúng kèm citation `nist-ai-rmf-1.0:p23:para:001`.
2. **Lỗi dev #2 — `q07` trong `cfg2_hybrid_4000_k5` (từ chối sai dù đã lấy được context liên quan):**
   - *Hiện tượng & trace:* Ở `cfg2`, agent chạy 3 lượt (`hybrid`, `hybrid`, `bm25`), trong `context_ids` đã có `nist-ai-rmf-1.0:p34:para:002`, `p41:para:001`, `p33:para:003` (`context_evidence = 15/15`), nhưng `answerable_answered` lại tụt từ `15/15` (`cfg1`) xuống `14/15` do model gọi `abstain`.
   - *Nguyên nhân:* Nhiều đoạn cùng nạp vào budget 4000 theo score chưa kéo được `p33:para:002`, và model thiếu hướng dẫn rà soát lại phần còn thiếu qua truy vấn phân tách trước khi kết luận thiếu bằng chứng.
   - *Khắc phục & kết quả:* Trong `cfg4_hybrid_4000_k5_followup`, hệ thống giữ hướng dẫn follow-up search có mục tiêu và cơ chế tự sửa citation (`citation_retries <= 1`). Kết quả `q07` được giải quyết trong 2 lượt tìm (bổ sung `p33:para:002`, trích dẫn `p33:para:002`, `p33:para:003`, `p34:para:002`), đưa `cfg4` lên **15/15 `context_evidence`**, **15/15 `answerable_answered`**, **2/2 `unanswerable_abstained`**, với median latency thấp hơn `cfg2` (`4646.64 ms` so với `4676.86 ms`) và chi phí thấp hơn (`$0.00895` so với `$0.009577`). Do đạt chuẩn toàn diện trên `dev`, `cfg4_hybrid_4000_k5_followup` được chọn làm cấu hình cuối.

---

## 3. Agent và đánh giá

### Kiến trúc tool, điều kiện dừng, chọn context và xử lý lỗi ([`src/agent.py`](../../src/agent.py))
- **Tools cung cấp cho `gpt-6-luna`:**
  1. `search_documents(query, mode, domain?, year?)`: hỗ trợ 3 chế độ `hybrid` / `dense` / `bm25`; khi `allow_agent_filter=true`, hợp nhất với `user_filters` (nếu có) và kiểm tra khóa hợp lệ qua `ALLOWED_FILTER_COLUMNS`.
  2. `submit_answer(answer, citations)`: trả lời kèm danh sách `chunk_id` trích dẫn và đánh dấu `[chunk_id]` trong văn bản.
  3. `abstain(reason)`: từ chối khi xác nhận tài liệu không chứa đủ bằng chứng.
- **Số lượt gọi & điều kiện dừng:** Dừng ngay khi model gọi `submit_answer` hợp lệ hoặc `abstain`; ép phải gọi `search_documents` ít nhất 1 lần trước khi trả lời/từ chối; giới hạn `max_searches` (chặn gọi vượt hạn mức bằng thông báo tool error trả lại `packed_context_ids`); giới hạn vòng lặp tối đa `max_turns = max_searches + 3`; nếu vượt hoặc model không gọi structured tool sẽ chuyển sang `abstain`.
- **Cách chọn context (`select_context`):** Ở baseline dùng `first_come` với `budget=800`; ở cấu hình cải thiện dùng `score_ranked` (khử trùng lặp theo `chunk_id`, chuẩn hóa điểm giữa các chế độ `hybrid`/`dense`/`bm25` để ưu tiên độ phù hợp cao nhất trong ngưỡng `4000` ký tự).
- **Xử lý lỗi:** API OpenAI Embeddings/Chat tự retry tối đa 3 lần khi gặp lỗi `429` hoặc `5xx`; kiểm tra toàn bộ `citations` do model trả về phải có mặt trong `final_context_ids` (cho phép tối đa 1 lần self-correction nếu có citation ngoài context, nếu vẫn vi phạm sẽ tự động chuyển về `abstain`).

### Bảng tổng hợp metrics Agent (`dev` tự động và `test` đã audit qua `starter/agent_metrics.py`)

| Cấu hình / split | Tool use | Phân bố số lượt gọi search | Grounded answer `0 / 1 / 2` (mean) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm (`false_abstention`) |
|---|---|---|---|---|---|---|---|
| Baseline `cfg0` / `dev` *(tự động)* | `17/17` | `{1: 6, 2: 11}` (avg `1.65`) | *Chưa chấm tay `dev`* (`answered=9/15`) | *Chưa chấm tay* | `9/9` | `2/2` | `6/15` |
| Cải thiện 1 `cfg1` / `dev` *(tự động)* | `17/17` | `{1: 11, 2: 5, 3: 1}` (avg `1.41`) | *Chưa chấm tay `dev`* (`answered=15/15`) | *Chưa chấm tay* | `15/15` | `2/2` | `0/15` |
| Cải thiện 2 `cfg2` / `dev` *(tự động)* | `17/17` | `{1: 11, 2: 2, 3: 4}` (avg `1.59`) | *Chưa chấm tay `dev`* (`answered=14/15`) | *Chưa chấm tay* | `14/14` | `2/2` | `1/15` |
| Chọn `cfg4` / `dev` *(tự động)* | `17/17` | `{1: 10, 2: 5, 3: 1, 4: 1}` (avg `1.59`) | *Chưa chấm tay `dev`* (`answered=15/15`) | *Chưa chấm tay* | `15/15` | `2/2` | `0/15` |
| **Baseline `cfg0` / `test` *(audit)*** | **`7/7`** | `{2: 7}` (avg `2.00`) | `5 / 0 / 0` (**`0.0 / 2.0`**) | `0/0` *(không áp dụng)* | `0/0` | **`2/2`** | `5/5` |
| **Đã chọn `cfg4` / `test` *(audit)*** | **`7/7`** | `{1: 2, 2: 2, 3: 2, 4: 1}` (avg `2.29`) | `0 / 2 / 3` (**`1.6 / 2.0`**) | **`9/9` (100%)** | **`6/6` (100%)** | `1/2` | **`0/5`** |

### Bảng audit chi tiết toàn bộ 7 câu holdout `test` — Cấu hình đã chọn (`cfg4_hybrid_4000_k5_followup`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn (`context` / `answer` / `cit` / `abstention`) & lý do |
|---|---|---|---|---|
| `q16` | Có đáp án (AI RMF resilience vs security) | 2 lượt (`hybrid`, `bm25`; `domain='ai_risk'`), context gồm `p20:para:002`, `p08:para:001`, `p26:para:002`, `p20:para:005` | Trình bày phân biệt resilience và security; mảng `citations`: `['nist-ai-rmf-1.0:p20:para:002']` | `ctx=1`, `ans=1`, `cit=[True]`: `p20:para:002` hỗ trợ phần graceful degradation & phân biệt resilience/security; văn bản có nhắc `[p20:para:003]` nhưng `p20:para:003` bị đẩy khỏi `final_context_ids` và mảng `citations` chỉ giữ `p20:para:002`. |
| `q17` | Có đáp án (AI RMF sau MAP) | 1 lượt (`hybrid`, no filter), context gồm `p33:para:001`, `p30:para:001`, `p35:para:001`, `p26:para:002` | Nêu kết quả MAP làm nền tảng cho MEASURE/MANAGE và quyết định ban đầu về tính phù hợp; `citations`: `['nist-ai-rmf-1.0:p30:para:001']` | `ctx=1`, `ans=1`, `cit=[True]`: `p30:para:001` hỗ trợ chính xác phần kết luận nêu ra, nhưng thiếu đoạn `p30:para:003` về quyết định `go/no-go`. |
| `q18` | Có đáp án (Appendix B AI risks) | 4 lượt (`hybrid`, `bm25`, `hybrid`, `bm25`; `domain='ai_risk', year=2023`), lấy được `p43:para:003`, `p43:para:004`, `p44:para:003` | Liệt kê các rủi ro đặc thù AI (quy mô, pre-trained bias/reproducibility, dữ liệu, tác động xã hội); `citations`: 3 ID trên | `ctx=2`, `ans=2`, `cit=[True, True, True]`: Cả 3 đoạn trích đều nằm trong context cuối và hỗ trợ nguyên văn toàn bộ luận điểm. |
| `q19` | Có đáp án (GOVERN 1.6 & 1.7) | 2 lượt (`hybrid`, `dense`; `domain='ai_risk', year=2023`), lấy được `p28:para:001` | Nêu đúng cơ chế inventory (GOVERN 1.6) và quy trình decommissioning/phase-out an toàn (GOVERN 1.7); `citations`: `['nist-ai-rmf-1.0:p28:para:001']` | `ctx=2`, `ans=2`, `cit=[True]`: Trúng đoạn chuẩn cả về nội dung lẫn nguồn trích dẫn. |
| `q20` | Có đáp án (CSF RC.RP-03) | 1 lượt (`hybrid`; `domain='cybersecurity', year=2024`), lấy được `p28:para:001` | Nêu đúng yêu cầu kiểm tra tính toàn vẹn backup/tài sản phục hồi trước khi dùng; `citations`: `['nist-csf-2.0:p28:para:001']` | `ctx=2`, `ans=2`, `cit=[True]`: Đúng và đủ trực tiếp theo Appendix A Core của NIST CSF 2.0. |
| `u_test_01` | Thiếu bằng chứng (biểu phí chứng nhận NIST CSF 2.0) | 3 lượt (`hybrid`, `bm25`, `dense`; `domain='cybersecurity'`), context gồm `p05:para:001`, `p08:para:001`, `p20:para:001`, `p06:para:003`, `p13:para:002`, `p19:para:001` | Gọi `submit_answer` (`status="answered"`): giải thích CSF 2.0 không nêu biểu phí chứng nhận; `citations`: `['nist-csf-2.0:p05:para:001', 'nist-csf-2.0:p08:para:001']` | `abstention=False`, `ans=0`, `cit=[True, True]`: Hai citation hỗ trợ khẳng định CSF mang tính outcome-based/non-prescriptive, nhưng về hành vi agent đã gọi `submit_answer` thay vì `abstain` trên câu hỏi ngoài phạm vi. |
| `u_test_02` | Thiếu bằng chứng (GDPR Art. 22 trong AI RMF) | 3 lượt (`hybrid`, `bm25`, `dense`), context các trang 6, 8, 23, 44–47 của AI RMF | Gọi `abstain`: xác nhận các đoạn AI RMF truy xuất không đề cập GDPR Article 22 hay mức phạt; `citations`: `[]` | `abstention=True`, `ans=0`, `cit=[]`: Tìm kiếm nhiều chiến lược rồi từ chối đúng quy định. |

*(Đối chiếu ngắn Baseline `cfg0` trên `test`: chạy `2` lượt search cho cả `7/7` câu nhưng do `budget=800` + `first_come` + chỉ `dense` mặc định, context chỉ giữ được 1 phần ở `q16` [`ctx=1`] và mất evidence ở `q17`–`q20` [`ctx=0`], dẫn đến từ chối nhầm `5/5` câu có đáp án [`mean=0.0`] và đúng `2/2` câu thiếu bằng chứng.)*

### Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí
- **Dữ liệu & kết quả chính:**
  - Retrieval `dev` / `test`: [`starter/results/dev-retrieval-benchmark/summary.json`](../../starter/results/dev-retrieval-benchmark/summary.json), [`starter/results/test-retrieval-benchmark/summary.json`](../../starter/results/test-retrieval-benchmark/summary.json)
  - Thử nghiệm Agent `dev`: [`starter/results/dev-agent-experiments/comparison.json`](../../starter/results/dev-agent-experiments/comparison.json)
  - Trace, audit & tổng hợp `test` Baseline: [`starter/results/test-agent-baseline/traces.json`](../../starter/results/test-agent-baseline/traces.json), [`reviews.json`](../../starter/results/test-agent-baseline/reviews.json), [`summary.json`](../../starter/results/test-agent-baseline/summary.json)
  - Trace, audit & tổng hợp `test` Selected (`cfg4`): [`starter/results/test-agent-selected/traces.json`](../../starter/results/test-agent-selected/traces.json), [`reviews.json`](../../starter/results/test-agent-selected/reviews.json), [`summary.json`](../../starter/results/test-agent-selected/summary.json)
- **Lệnh chạy tái lập:**
  ```powershell
  .venv\Scripts\python.exe -m unittest discover -s tests -v
  .venv\Scripts\python.exe -m src.benchmark_retrieval --split dev --output starter/results/dev-retrieval-benchmark
  .venv\Scripts\python.exe -m src.run_dev_experiments
  .venv\Scripts\python.exe -m src.run_test_evaluation
  .venv\Scripts\python.exe starter\agent_metrics.py --cases starter\data\converted\agent_cases.json --traces starter\results\test-agent-selected\traces.json --reviews starter\results\test-agent-selected\reviews.json --split test --output starter\results\test-agent-selected\summary.json
  ```
- **Thời gian & chi phí thực tế (`gpt-6-luna` + `text-embedding-3-small`):**
  - Toàn bộ 4 cấu hình thử nghiệm trên `dev` (17 cases/cấu hình): `provider_errors = 0`; chi phí từ `$0.005733` (`cfg0`) đến `$0.009577` (`cfg2`), riêng cấu hình chọn `cfg4` hết `$0.00895` với median `4646.64 ms/case`.
  - Holdout `test` (7 cases): Baseline `cfg0` hết `$0.002561` (median `6350.26 ms`); Selected `cfg4` hết `$0.0059` (median `9070.90 ms`).

### Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn
- **Cải thiện trước/sau trên holdout `test`:** So với `cfg0_baseline_dense_800`, cấu hình chọn `cfg4_hybrid_4000_k5_followup` nâng `grounded_answer.mean` từ **`0.0` lên `1.6 / 2.0`**, giảm `false_abstention` từ **`5/5` xuống `0/5`**, đạt **`9/9` (100%) citation hợp lệ** và **`6/6` câu trả lời có citation**.
- **Ví dụ minh họa trên `test`:**
  - *Trả lời có bằng chứng đạt điểm tối đa:* `q19` và `q20` đều lấy trúng đoạn chuẩn (`nist-ai-rmf-1.0:p28:para:001` và `nist-csf-2.0:p28:para:001`), trả lời ngắn gọn, bám sát nguyên văn và trích dẫn chính xác.
  - *Từ chối đúng trên câu thiếu bằng chứng:* `u_test_02` chạy đủ 3 chế độ tìm kiếm (`hybrid`, `bm25`, `dense`) để kiểm chứng trước khi gọi `abstain`.
  - *Lỗi quan sát được trên `test`:* Ở `u_test_01`, `gpt-6-luna` tìm thấy đoạn mô tả CSF mang tính tự nguyện (`p05:para:001`, `p08:para:001`) nên dùng `submit_answer` để kết luận tài liệu không quy định biểu phí thay vì gọi `abstain` (làm `abstention` chỉ đạt `1/2`). Ngoài ra ở `q16` và `q17`, `select_context` ưu tiên điểm của các đoạn khác nên chưa giữ lại đoạn bổ trợ (`p20:para:003` ở `q16` và `p30:para:003` ở `q17`), khiến 2 câu chỉ đạt mức `1`.
- **Đánh đổi:** Tăng `top_k` (`3 -> 5`), `max_searches` (`2 -> 4`) và `budget` (`800 -> 4000`) giúp loại bỏ hoàn toàn tình trạng mất bằng chứng và từ chối nhầm, nhưng làm chi phí trên `test` tăng khoảng `2.3x` (`$0.002561 -> $0.0059`) và median latency tăng từ `6.35 s` lên `9.07 s`.
- **Giới hạn & hướng phát triển:**
  1. Quy mô benchmark còn gọn (`17` dev, `7` test) trên 2 tài liệu NIST tiếng Anh (`67` trang được index).
  2. Chưa có cơ chế nạp thêm đoạn liền kề (`neighbor chunk expansion`, ví dụ tự kèm `p20:para:003` hoặc `p30:para:003`) khi giữ theo `score_ranked`.
  3. Cần làm rõ hơn trong schema/prompt ranh giới giữa *"tài liệu khẳng định không có chứng nhận"* và *"câu hỏi ngoài phạm vi phải gọi `abstain`"* (như lỗi `u_test_01`) ở vòng phát triển tiếp theo trên tập `dev` độc lập.

---

## 4. Thử thách nâng cao (Tài liệu 124 trang & Multimodal)

### 4.1. Tài liệu thực tế, quyền sử dụng và chiến lược index đa phương thức
- **Tài liệu nguồn:** *The Global Risks Report 2024 (19th Edition)*, World Economic Forum (WEF), xuất bản tháng 1/2024, tải công khai tại [`https://www3.weforum.org/docs/WEF_The_Global_Risks_Report_2024.pdf`](https://www3.weforum.org/docs/WEF_The_Global_Risks_Report_2024.pdf) (**124 trang**, `13,194,848` bytes, `sha256: af3939585e44d9ccf8d2f5d3d67bf696c1f84f44aac0f711878f515671aee593`). File PDF gốc và ảnh render nặng được loại khỏi git trong `.gitignore`; người chấm tái tạo đầy đủ qua [`src/advanced/download_doc.py`](../../src/advanced/download_doc.py) và [`src/advanced/build_index.py`](../../src/advanced/build_index.py).
- **Quy mô index** ([`data/advanced/index/manifest.json`](../../data/advanced/index/manifest.json)):
  - Toàn bộ 124 trang được xử lý tạo **648 chunks** lưu trong PostgreSQL + `pgvector` (bảng `advanced_chunks`, embedding `text-embedding-3-small` 1536 chiều): **`540` text chunks**, **`12` table chunks**, **`96` visual chunks** (`visual_pages_indexed = 96`).
  - Mỗi chunk giữ `source_id`, `page`, `bbox`, `figure_id`, `page_image_path` (render toàn trang 140 DPI tại `data/advanced/index/pages/pXXX.png`) và `crop_path` (vùng bảng/biểu đồ/sơ đồ 180 DPI tại `data/advanced/index/crops/pXXX_visual.png`) để đối chiếu trực tiếp bản gốc.
- **Trang bỏ qua, lý do và ảnh hưởng:**
  - `pages_without_text`: `[1, 124]` (bìa trước và bìa sau).
  - `skipped_visual_pages` (28 trang): `cover_or_imprint` (`1, 2, 124`), `text_only_page` (`3–5, 10, 84, 115–123`), và `decorative_stock_photo_only` (`20, 24, 26, 30, 32, 41, 66, 85, 87, 90, 92` — chỉ có ảnh Unsplash trang trí không mang số liệu; toàn bộ 124 trang vẫn có PNG toàn trang nếu cần mở).
- **Kiến trúc tìm kiếm & Vision agentic ([`src/advanced/retriever.py`](../../src/advanced/retriever.py), [`src/advanced/agent.py`](../../src/advanced/agent.py)):**
  - `AdvancedHybridRetriever` hỗ trợ 3 chế độ `hybrid` (RRF `pgvector` Dense + Okapi BM25), `dense`, `bm25` kèm bộ lọc `modality` (`text`/`table`/`visual`) và `page_start..page_end`.
  - Do văn bản rút từ bảng nhiều cột (`p103:table:01`) hoặc bản tóm tắt/crop hẹp có thể trộn cột hoặc cắt lề nhãn, pipeline cung cấp tool `inspect_page_image(page, focus)` gắn ảnh PNG trang gốc độ phân giải cao (`detail="high"`) trực tiếp vào hội thoại của `gpt-6-luna`, giúp kiểm chứng con số, trục tọa độ, chú giải và liên kết trên bản gốc trước khi gọi `submit_answer`.
- **Chi phí & thời gian dựng index:** `129.06 s`, `138,433` embedding tokens (`$0.002769`), `275,165` vision prompt tokens + `35,528` completion tokens (`$0.045281`), tổng **`$0.04805`**.

### 4.2. Thiết kế Benchmark (`dev` & `test` khóa trước khi thử nghiệm)
- **Tập `dev` ([`data/advanced/benchmark/dev.json`](../../data/advanced/benchmark/dev.json), `18` câu):** `3` Bảng, `3` Ảnh/sơ đồ, `3` Biểu đồ, `3` Multi-hop, `3` Text, `3` Thiếu bằng chứng.
- **Tập `test` holdout ([`data/advanced/benchmark/test.json`](../../data/advanced/benchmark/test.json), `30` câu):** `6` Bảng, `5` Ảnh/sơ đồ, `5` Biểu đồ, `4` Multi-hop, `4` Text, `6` Thiếu bằng chứng (`24` có đáp án + `6` thiếu bằng chứng).
- Mọi câu `Ảnh, sơ đồ` và `Biểu đồ` đều nhắm vào sơ đồ, bản đồ dữ liệu, chuỗi giá trị, radar chart, line chart có nhãn kiểm chứng (không dùng ảnh trang trí). Kiểm tra không trùng ID/câu hỏi giữa `dev` và `test` qua unit test `test_benchmark_quotas_and_zero_overlap` trong [`tests/test_advanced.py`](../../tests/test_advanced.py).

### 4.3. Kết quả thử nghiệm trên `dev` và quá trình chỉnh lỗi trước khi khóa pipeline

| Cấu hình (`gpt-6-luna`) | Split | Đạt | Bảng | Ảnh/sơ đồ | Biểu đồ | Multi-hop | Text | Thiếu bằng chứng | Avg latency / Tổng chi phí |
|---|---|---|---|---|---|---|---|---|---|
| `cfg0_text_only_baseline` (`modality="text"`, no vision) | dev (18) | `14/18` (77.78%) | `2/3` | `2/3` | `3/3` | `2/3` | `3/3` | `2/3` | `6373.48 ms` / `$0.018107` |
| `cfg1_multimodal_no_live_vision` (tất cả chunk, không `inspect_page_image`) | dev (18) | `11/18` (61.11%) | `0/3` | `3/3` | `1/3` | `2/3` | `3/3` | `2/3` | `5818.99 ms` / `$0.024358` |
| **`cfg2_multimodal_live_vision` — Đã chốt** (`inspect_page_image` `focus="full"`, ưu tiên hit mới, kiểm phạm vi) | **dev (18)** | **`18/18` (100.0%)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `5481.22 ms` / `$0.020911` |

- **Hai lỗi phát hiện và sửa trên `dev` trước khi khóa pipeline:**
  1. *Cắt lề crop ở đồ thị ngang/cột (`dev-cht-02`, `dev-cht-03`):* Khi mặc định `focus="crop"`, bbox hẹp cắt mất nửa chữ nhãn bên trái (`crops/p088_visual.png` chỉ còn `"...er technologies"`, làm model đọc nhầm sang `AI technologies`) và cắt mất chú giải thiếu hụt `US$194–366B` bên phải (`crops/p046_visual.png`). Chuyển mặc định `inspect_page_image` sang `focus="full"` khắc phục triệt để cả hai câu.
  2. *Đóng gói ngược thứ tự khi search nhiều lượt (`dev-tbl-03`, `dev-mh-01`) & suy diễn ấn bản (`dev-una-01`):* `candidates.extend(res)` đặt kết quả lượt tìm 1 lên trước khiến hit ở lượt 2–3 (`p096`, `p015`) bị tràn `context_budget`. Sửa thành `candidates = list(res) + candidates` kết hợp danh sách `shown_ids` tích lũy và siết quy tắc kiểm tra đúng năm/ấn bản (`Global Risks Report 2024` vs `2025`), đưa `dev` lên **18/18 (100.0%)**.

### 4.4. Kết quả holdout `test` (chạy 1 lượt duy nhất, audit thủ công từng câu với PDF gốc)
- **Tổng kết ([`summary.json`](../../data/advanced/results/test/cfg2_multimodal_live_vision/summary.json), [`reviews.json`](../../data/advanced/results/test/cfg2_multimodal_live_vision/reviews.json)):**
  - **Độ chính xác chung:** **`28/30` (`93.33%`)** (vượt ngưỡng `90%` / `27/30`, không làm tròn; `0` lỗi hệ thống/timeout).
  - **Theo nhóm chính:** Bảng **`6/6` (100.0%)**, Ảnh/sơ đồ **`5/5` (100.0%)**, Biểu đồ **`5/5` (100.0%)**, Multi-hop **`3/4` (75.0%)**, Text **`3/4` (75.0%)**, Thiếu bằng chứng **`6/6` (100.0%)**.
  - **Thời gian & chi phí trên `test` (30 câu):** Tổng chi phí **`$0.041576`** (`$0.001386/câu`), thời gian trung bình **`5658.49 ms/câu`** (`p95 = 9185.61 ms`), `383,956` prompt tokens + `6,334` completion tokens + `651` embedding tokens, **`28`** lượt gọi `inspect_page_image`.
- **Phân tích 2 câu không đạt trên `test`:**
  1. `test-mh-03` *(Multi-hop — `p043`, `p044`)*: Đọc đúng ngưỡng nhiệt độ AMOC (`1.4°C` possible, `4.0°C` likely) trên Box 2.2 và Category `Circulation patterns`, nhưng trên bản đồ dày đặc `Figure 2.5` (p.43) nhận nhầm điểm bùng phát lân cận thứ 2 nhãn `Collapse` phía bắc/tây bắc AMOC là `Boreal permafrost` (thực tế nhãn phía tây bắc của `Boreal permafrost` ghi `Abrupt thaw`, còn điểm `Boreal permafrost - Collapse` nằm tận vùng Siberia phía đông bắc; hai điểm `Collapse` lân cận phía bắc/tây bắc AMOC là `Labrador sea/subpolar gyre` và `Greenland ice sheet`).
  2. `test-txt-03` *(Text — `p010` vs `p023`)*: Agent trả lời dựa vào đoạn tóm tắt tổng quan Chương 1 (`p010:text:02`: *"the Russia-Ukraine war, the Middle East conflict and tensions over Taiwan"*) mà chưa đọc tiếp trang 23 (nằm ở hạng 6 trong kết quả search) thuộc đúng tiêu đề *"High-stakes hotspots"* nơi nêu cụ thể *"the Israel-Gaza conflict"*.
- **Giới hạn & hướng cải thiện:** Với bản đồ mật độ nhãn dày như `Figure 2.5`, việc crop phóng to theo vùng địa lý khi zoom (`quadrant zoom`) sẽ tăng độ chính xác thị giác; với các truy vấn text có tên tiểu mục cụ thể, có thể bổ sung cơ chế ưu tiên trích thêm chunk đứng cuối top-k hoặc mở trang mục tiêu khi tiêu đề mục xuất hiện ở kết quả hạng 4–6.
- **Lệnh chạy tái lập Thử thách nâng cao:**
  ```powershell
  .venv\Scripts\python.exe -m unittest discover -s tests -v
  .venv\Scripts\python.exe -m src.advanced.download_doc
  .venv\Scripts\python.exe -m src.advanced.build_index
  .venv\Scripts\python.exe -m src.advanced.run_dev
  .venv\Scripts\python.exe -m src.advanced.run_test
  .venv\Scripts\python.exe -m src.advanced.score_results --traces data/advanced/results/test/cfg2_multimodal_live_vision/traces.json --reviews data/advanced/results/test/cfg2_multimodal_live_vision/reviews.json --out data/advanced/results/test/cfg2_multimodal_live_vision/summary.json
  ```
