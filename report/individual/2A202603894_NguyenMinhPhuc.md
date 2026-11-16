# Báo cáo cá nhân Day 7 — Nguyễn Minh Phúc (2A202603894)

## Dữ liệu và benchmark

- Kế thừa kho Day 6: 4 PDF nguồn, 221 trang nội dung được index (36 trang bỏ qua); index B có 714 chunk theo Điều. Chunk có source, page và metadata để truy vết citation. Quyền sử dụng: cùng quyền sử dụng nguồn đã ghi trong báo cáo Day 6; cần đính kèm/đối chiếu giấy phép nguồn khi nộp.
- PostgreSQL 16 + pgvector, index B; embedding `text-embedding-3-small`, 1536 chiều. Embedding câu hỏi dùng cùng model và chiều với vector đã lưu. Lexical TF-IDF chạy trên bản sao 714 chunk; hybrid hợp nhất dense và lexical bằng RRF (constant 60).
- Model agent: `gpt-5.4-mini` qua OpenAI Chat Completions tool calling; key đọc từ `.env` ở thư mục gốc repo. Agent nhận câu hỏi và context truy xuất, không nhận gold. Tối đa 4 lượt tìm; citation phải trỏ chunk trong context cuối và xuất hiện trong câu trả lời.
- Benchmark Day 7: dev 160 answerable + 2 câu cần abstain; test 40 answerable + 2 cần abstain. Câu answerable kế thừa từ benchmark Day 6; test Day 6 đã được báo cáo trước đó, nên không phải holdout mới hoàn toàn. Cấu hình được chọn chỉ dựa trên dev. Evidence Hit đo khớp source/page và evidence quote trong chunk; câu trả lời và citation support được review ngữ nghĩa (xem mục Đánh giá).

## Retrieval và thử nghiệm dev

| Cấu hình | Phương pháp / tham số | Giả thuyết hoặc thay đổi | Hit@1 | Evidence Hit@3 | Context evidence | Trả lời /160 | Latency TB / median | Chi phí ước tính |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF baseline | TF-IDF, top-3, context 800 ký tự | Mốc lexical; bắt từ pháp lý chính xác | 84/160 | 103/160 (64.4%) | 45/160 (28.1%) | 19 | — | $0.3510 |
| Dense baseline | pgvector cosine, top-3, context 800 | Bắt paraphrase ngữ nghĩa | 87/160 | 122/160 (76.3%) | 42/160 (26.3%) | 17 | — | $0.3623 |
| Hybrid RRF baseline | dense + TF-IDF, 20 ứng viên/nguồn, top-3, context 800 | Kết hợp exact terms và semantic | 103/160 | 139/160 (86.9%) | 51/160 (31.9%) | 21 | — | $0.3476 |
| Cải thiện 1: wide candidates | RRF, 40 ứng viên/nguồn, top-3, context 800 | Tăng recall ở pool ứng viên | 102/160 | 137/160 (85.6%) | 53/160 (33.1%) | 21 | — | $0.3427 |
| Cải thiện 2: more context | RRF, 20 ứng viên/nguồn, top-5, context 1,400 | Đưa thêm bằng chứng vào context cuối | 102/160 | 135/160 (84.4%) | 101/160 (63.1%) | 27 | 4.41 s / 3.13 s | $0.3116 |
| **Cải thiện 3: citation fix (đã chọn)** | **Retrieval giống more context; chuẩn hóa citation ID và bắt buộc marker `[id]` trong câu trả lời** | **Phần lớn abstain nhầm do lỗi định dạng citation, không phải thiếu evidence** | **106/160** | **139/160 (86.9%)** | **105/160 (65.6%)** | **133** | **3.50 s / 2.76 s** | **$0.3323** |

Chi phí là ước lượng theo token usage và giá cấu hình, không phải hóa đơn. Latency của các run đầu nằm trong `summary.json` từng cấu hình. Retrieval của "citation fix" giống "more context" nên chênh lệch Hit nhỏ (+4 Hit@1/Hit@3, +4 context evidence) đến từ khác biệt truy vấn tìm kiếm của agent giữa hai lần chạy, không phải thay đổi retriever.

### Phân tích lỗi và lý do chọn

- **Lỗi chính của "more context":** abstain nhầm 133/160 câu có đáp án dù context cuối chứa evidence ở 101/160 câu. Giả thuyết cải thiện 3 (ghi trong config): model trả citation ID không ở dạng chuẩn hoặc thiếu marker `[id]` trong câu trả lời, nên bộ kiểm tra citation loại bỏ câu trả lời hợp lệ về cấu trúc → abstain. Vì vậy chỉ sửa chuẩn hóa citation, giữ nguyên retrieval để cô lập nguyên nhân.
- **Kết quả:** số câu trả lời tăng 27 → 133; abstain nhầm giảm 133 → 27; abstention đúng giữ 2/2. Câu đúng (grade 2) tăng 24 → 114; grade trung bình trên 160 câu tăng 0.30 → 1.48. Chi phí tăng nhẹ $0.3116 → $0.3323 (+6.6%), latency trung bình giảm 4.41 s → 3.50 s.
- **Chọn `hybrid_rrf_citation_fix`** trên dev: cao nhất hoặc ngang cao nhất về Evidence Hit@3 và Context evidence, vượt trội về câu trả lời/citation sau review, chi phí tương đương. Không dùng test để chọn.

Lỗi dev còn lại của cấu hình đã chọn (19 câu grade < 2, xem `review_note` trong queue):

- **Dẫn sai điều luật dù đúng chủ đề** (`q104`, `q124`, `q126`, `q051`, `q195`): model chọn thời hạn ở điều lân cận (vd. `q104` trả 10 ngày của thủ tục thu hồi giấy phép thay vì thời hạn thẩm tra hồ sơ cấp phép; `q124` trả 30 ngày thay vì báo trước 03 ngày làm việc khi chậm lương). Nguyên nhân: context top-5 chứa nhiều điều có cấu trúc "trong thời hạn N ngày" và model chọn nhầm.
- **Dùng BLLĐ 2012 thay cho BLLĐ 2019** (`q060`, `q078`, `q179`): retrieval trả chunk luật cũ cùng nội dung; câu trả lời mơ hồ hoặc sai cơ quan có thẩm quyền. Hướng thử tiếp: filter/ưu tiên văn bản còn hiệu lực khi câu hỏi không nêu năm.
- **Trả lời "không đủ căn cứ" khi có evidence ở điều khác** (`q007`, `q087`, `q192`): retrieval không đưa được điều chứa đáp án vào context.
- **Thiếu điều kiện** (`q035` thiếu "bằng văn bản", `q116` thiếu công thức tính ngày phép, `q096`, `q185`, `q190`, `q031` mâu thuẫn nội bộ).

## Agent và đánh giá

Tool `search_documents` trả top-k chunk kèm source/page/score/rank; controller quyết định tìm tiếp hoặc `submit_answer`. Agent tích lũy context trong giới hạn ký tự, dừng sau tối đa 4 lần search; thiếu citation hợp lệ trong context cuối thì abstain.

| Cấu hình dev | Answer grade 0 / 1 / 2 (trên 160) | Trung bình | Citation hỗ trợ / citation được chấm | Câu trả lời có mọi citation hỗ trợ | Đúng + citation hỗ trợ đầy đủ | Abstention đúng | Abstain nhầm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid RRF baseline | 141 / 1 / 18 | 0.23 | 20/22 | — | 18 | 2/2 | 139 |
| Hybrid more context | 136 / 0 / 24 | 0.30 | 24/27 | 24/27 | 24 | 2/2 | 133 |
| **Hybrid citation fix (đã chọn)** | **37 / 9 / 114** | **1.48** | **127/137** | **123/133** | **111** | **2/2** | **27** |

Grade 0 bao gồm câu abstain nhầm. Tất cả 133 citation ID của "citation fix" có trong context cuối, nhưng review ngữ nghĩa chỉ xác nhận 127/137 citation thực sự hỗ trợ câu trả lời; một số câu đúng nhưng cite điều lân cận (vd. `q039` cite Điều 36 thay vì Điều 35, `q125` cite Điều 37 thay vì Điều 38).

**Cách review:** 133 câu trả lời của "citation fix" được review bằng LLM (AI-assisted): mỗi câu đối chiếu answer với expected answer, gold evidence và text/page của chunk được cite; tiêu chí 2 = đúng và đủ, 1 = đúng một phần/thiếu điều kiện/sai phiên bản luật, 0 = sai hoặc không trả lời. Kết quả nằm trong `citation_review_queue.json` (trường `answer_grade`, `citation_support`, `abstention_correct`, `review_note`, mỗi note gắn nhãn AI-assisted). Review của các cấu hình trước cũng được thực hiện theo cùng tiêu chí. **Giới hạn:** nhãn chưa được người kiểm tra lại toàn bộ; cần spot-check trước khi coi là nhãn cuối.

Artifacts dev:

- `artifacts/day7_dev/run_20261116/` — 5 cấu hình đầu, `selection.md`, trace/score/review queue.
- `artifacts/day7_dev/run_20261116_citation_fix/hybrid_rrf_citation_fix/` — trace, score, review queue đã chấm, `summary.json` (có `semantic_answer_grades`).

## Test

Quy trình đánh giá test: chốt cấu hình trên dev rồi chạy

```bash
.venv/bin/python -m starter.test_experiments --output artifacts/day7_test/<run_id>
```

**Lưu ý về thứ tự thực hiện (trung thực):** `artifacts/day7_test/run_20261116_final/` được tạo lúc 23:24 ngày 16/11 cho hybrid baseline và `hybrid_rrf_more_context` (cấu hình chốt khi đó). Cấu hình "citation fix" được thử trên dev sau đó (23:42–23:51) dựa trên phân tích lỗi abstain trên dev; kết quả test của run 23:24 **không được đọc hay dùng** khi chọn "citation fix". Cấu hình "citation fix" chưa chạy trên test; cần thêm `hybrid_rrf_citation_fix` vào `CONFIG_NAMES` trong `starter/test_experiments.py`, chạy test một lần, review queue test và điền bảng dưới.

| Cấu hình / test | Hit@1 | Evidence Hit@3 | Context evidence | Grade 0/1/2 | Citation hỗ trợ | Abstention đúng | Thời gian / chi phí |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid RRF baseline | | | | | | | |
| Hybrid citation fix (đã chọn) | | | | | | | |

## Kết luận, đánh đổi và giới hạn

- Hybrid RRF cải thiện retrieval rõ so với từng phương pháp đơn lẻ (Evidence Hit@3 139 vs 122 dense, 103 TF-IDF). Tăng context budget gần như gấp đôi evidence trong context cuối. Cải thiện lớn nhất cho chất lượng trả lời đến từ việc sửa xử lý citation (27 → 133 câu trả lời), cho thấy nút thắt nằm ở controller/validator chứ không phải retriever.
- Đánh đổi: context 1,400 ký tự và nhiều câu trả lời hơn làm chi phí tăng nhẹ (+6.6%) trong khi latency trung bình không tăng (4.41 s → 3.50 s; chưa phân tích số lượt search để giải thích nguyên nhân).
- Giới hạn: benchmark kế thừa từ Day 6 nên test không phải holdout mới; nhãn answer/citation là AI-assisted, cần người spot-check; chi phí là ước tính theo token; còn 27 câu abstain nhầm và lỗi chọn nhầm điều/phiên bản luật (BLLĐ 2012 vs 2019).
- Chưa làm thử thách nâng cao.
