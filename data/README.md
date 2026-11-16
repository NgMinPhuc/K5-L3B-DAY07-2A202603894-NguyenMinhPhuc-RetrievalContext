# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

Benchmark Day07 hiện được chuẩn bị tại [benchmark_day7.json](benchmark_day7.json): 160 câu có đáp án + 2 câu thiếu evidence trên dev, 40 câu có đáp án + 2 câu thiếu evidence trên test. Chạy `python3 starter/prepare_day7_benchmark.py` để tái tạo từ benchmark Day06 đã sao chép và [abstention_cases.json](abstention_cases.json). Câu có đáp án giữ evidence/source/page và các alternative gold của Day06; câu thiếu evidence ghi lý do vắng mặt. Day06 đã công bố kết quả trên 40 câu test này, nên không dùng test để chọn cấu hình Day07; nếu kết quả test ảnh hưởng đến điều chỉnh agent, hãy tạo một test holdout mới.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).
