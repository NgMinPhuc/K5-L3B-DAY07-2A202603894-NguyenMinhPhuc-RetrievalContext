# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Ba phương pháp retrieval đã cài

`starter/retrieval.py` cung cấp giao diện chung `search(query, filters=None, top_k=3)` và trả chunk có `id`, `text`, nguồn/trang, metadata, score/rank:

| Phương pháp | Cách chạy | Cấu hình hiện tại |
|---|---|---|
| Dense | `retrievers["dense"]` | Index B của Day06, pgvector cosine, query embedding `text-embedding-3-small` (1536 chiều) |
| Lexical | `retrievers["tfidf"]` | TF-IDF trên 714 chunk B đã sao chép; không gọi API |
| Hybrid | `retrievers["hybrid_rrf"]` | Lấy 20 ứng viên từ mỗi retriever, hợp nhất bằng Reciprocal Rank Fusion (RRF), hằng số 60 |

Ví dụ khởi tạo từ thư mục gốc; chỉ gọi `search` trên dense/hybrid mới cần `OPENAI_API_KEY` và PostgreSQL:

```python
from pathlib import Path
from starter.retrieval import build_retrievers

retrievers = build_retrievers(Path("starter/day6/data/chunks_B.jsonl"))
hits = retrievers["tfidf"].search("Thời gian thử việc tối đa là bao lâu?", top_k=3)
# Để dùng dense: hits = retrievers["dense"].search(query, top_k=3)
# Hybrid: hits = retrievers["hybrid_rrf"].search(query, top_k=3)
```

Ba phương pháp này khác cách lấy/xếp hạng ứng viên; thay top-k riêng lẻ không được tính là phương pháp mới. Filter whitelist của dense là `year`, `document_id`, `source_id`, `document_type`; TF-IDF cũng áp filter trên metadata. Chỉ truyền filter từ nội dung câu hỏi/ngữ cảnh đã xác định, không lấy từ gold. Kết quả hybrid lưu thêm `rrf_ranks` để truy vết thứ hạng của dense và TF-IDF.

## 3. Model thật, trace và citation review

Cài dependencies và tạo cấu hình local (không commit `.env`):

```bash
python3 -m pip install -r starter/requirements.txt
cp .env.example .env
```

Điền `OPENAI_API_KEY` vào `.env` ở thư mục gốc repo. Mẫu chọn `gpt-5.4-mini` qua Chat Completions tool calling; có thể đổi `OPENAI_CHAT_MODEL` sang model khác hỗ trợ function calling. Không chia sẻ file `.env`. Bộ điều khiển là `starter/model_controller.py`; nó chỉ nhận câu hỏi và context retrieval, không nhận `gold` hay đáp án chuẩn.

`starter/trace_store.py` ghi JSONL theo từng câu, gồm tool calls, kết quả, context cuối, câu trả lời, citation, cấu hình, thời gian và lỗi. Trace không lưu gold. `starter/citation_review.py` tạo hàng đợi chấm thủ công với text/source/page của các chunk được cite và answer/evidence chuẩn chỉ để review offline. Công cụ chỉ đánh dấu citation ID có trong context; người chấm vẫn phải xác nhận citation có hỗ trợ nội dung câu trả lời.

## 4. Thử cấu hình chỉ trên dev

Chạy từ thư mục gốc sau khi cấu hình model/API:

```bash
python3 -m starter.dev_experiments
```

Nếu tiến trình dừng giữa chừng, tiếp tục cùng thư mục output bằng `python3 -m starter.dev_experiments --output artifacts/day7_dev/<timestamp> --resume`; runner bỏ qua các ID đã có trace.

Runner chỉ đọc `data/dev_benchmark_day7.json` (162 câu dev), không mở benchmark test. Nó chạy TF-IDF, dense, hybrid RRF và hai cấu hình cải thiện: candidate pool hybrid 20 → 40; top-k/context 3/800 → 5/1400 ký tự. Kết quả mặc định vào `artifacts/day7_dev/<timestamp>/`, có trace JSONL không chứa gold, `scores.jsonl` cho đánh giá retrieval offline, summary theo cấu hình và citation review queue. Gold chỉ được đọc sau khi agent trả lời để tính retrieval metrics hoặc tạo review queue.

Đặt `OPENAI_INPUT_PRICE_PER_1M_USD` và `OPENAI_OUTPUT_PRICE_PER_1M_USD` trong `.env` để ước lượng chi phí chat từ token usage. Nếu không có giá, runner để chi phí chat là null. Đọc citation review queue, điền `answer_grade`, `citation_support` và `abstention_correct`, đọc ít nhất hai lỗi dev, rồi chọn cấu hình theo Evidence Hit@3, Context Evidence Rate, câu trả lời/citation sau review, thời gian và chi phí. Chưa chạy test cho đến khi chốt cấu hình.

Sau khi chốt cấu hình trên dev, chạy baseline và cấu hình đã chọn trên test bằng runner riêng:

```bash
.venv/bin/python -m starter.test_experiments --output artifacts/day7_test/<timestamp>
```

Runner chỉ đọc `data/test_benchmark_day7.json`, chạy `hybrid_rrf_baseline` và `hybrid_rrf_more_context`, rồi ghi trace, scores, summary và review queue riêng. Có thể tiếp tục sau lỗi bằng `--resume`. Test Day 7 kế thừa các câu answerable đã dùng trong báo cáo Day 6, nên xem đây là so sánh cấu hình trên split được giữ riêng cho lựa chọn Day 7, không phải holdout mới hoàn toàn.

## 5. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 6. Dùng dữ liệu Day 6

Trong repo này đã có bản sao các artifact cần cho Day 7 tại `starter/day6/`:

- `data/chunks_B.jsonl`: 714 chunk theo Điều, gồm nội dung, ID, trang và metadata.
- `data/questions.json`: benchmark Day 6 (200 câu, 160 dev / 40 test).
- `src/`: adapter embedding OpenAI và pgvector của Day 6; index dense vẫn nằm trong PostgreSQL Day 6.
- `results/`: metrics và thống kê chunk để tham khảo.

Không copy `.env` từ project Day 6 bên ngoài repo. File `.env` riêng của lab nằm ở thư mục gốc repo; chỉ điền `OPENAI_API_KEY` của bạn và giữ `EMBEDDING_MODEL=text-embedding-3-small`. Không sao chép key từ project khác. `DATABASE_URL` trỏ tới PostgreSQL local (`127.0.0.1:55433`, database `day6_lab`); đây là thông tin kết nối local, không phải API secret. Nếu service Day 6 đang chạy, dùng lại service đó; nếu chưa, khởi động compose từ `starter/day6/` bằng project name Day 6 để nối lại cùng volume. Không chạy hai Postgres cùng lúc trên cổng này.

Embedding câu hỏi phải dùng cùng model và số chiều với index: Day 6 đã dùng `text-embedding-3-small`, 1536 chiều. Đổi model sẽ không tương thích với vector đã lưu; cần tạo index mới. PostgreSQL Day 6 đã được xác nhận chạy, có 788 vector A và 714 vector B; truy vấn cosine trên index B thành công. Khi chạy agent, kiểm tra `EMBEDDING_MODEL` trong `.env` vẫn là `text-embedding-3-small`; cần `OPENAI_API_KEY` hợp lệ để gọi embedding API cho câu hỏi.

Benchmark Day 6 chỉ có câu có đáp án. Benchmark Day07 đã được tạo tại `data/benchmark_day7.json`: 160 câu có đáp án + 2 câu abstention trên dev, 40 câu có đáp án + 2 câu abstention trên test. Tái tạo bằng `python3 starter/prepare_day7_benchmark.py`. Evidence của 200 câu được giữ từ benchmark Day06; bốn câu abstention có lý do thiếu evidence trong `data/abstention_cases.json`. Day06 đã báo cáo kết quả trên 40 câu test; đừng dùng chúng để chọn cấu hình agent Day07. Nếu test ảnh hưởng đến thay đổi pipeline, hãy tạo test holdout mới.

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 7. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
