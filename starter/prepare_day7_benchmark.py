"""Convert the copied Day 6 benchmark and add reviewed abstention cases."""
import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6", type=Path, default=ROOT / "starter/day6/data/questions.json")
    parser.add_argument("--abstentions", type=Path, default=ROOT / "data/abstention_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark_day7.json")
    args = parser.parse_args()

    old = json.loads(args.day6.read_text(encoding="utf-8"))
    source_questions = old["questions"] if isinstance(old, dict) else old
    abstentions = json.loads(args.abstentions.read_text(encoding="utf-8"))
    questions = []
    for q in source_questions:
        gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"],
                 "evidence": q["evidence_quote"]}]
        gold.extend({"source_id": g["source_id"], "page": g["page"],
                     "evidence": g["evidence_quote"]} for g in q.get("alt_gold", []))
        item = {"id": q["id"], "split": q["split"], "question": q["question"],
                "answerable": True, "expected_answer": q["expected_answer"], "gold": gold}
        if q.get("filter"):
            item["filter"] = q["filter"]
        if q.get("type"):
            item["tags"] = [q["type"]]
        questions.append(item)

    questions.extend(abstentions)
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("ID câu hỏi bị trùng")
    counts = Counter((q["split"], q["answerable"]) for q in questions)
    required = {("dev", True): 15, ("dev", False): 2, ("test", True): 5, ("test", False): 2}
    for key, minimum in required.items():
        if counts[key] < minimum:
            raise ValueError(f"Không đủ câu {key}: {counts[key]} < {minimum}")
    for q in questions:
        if q["answerable"] and (not q.get("gold") or any(not g.get("evidence") for g in q["gold"])):
            raise ValueError(f"Thiếu evidence: {q['id']}")

    result = {
        "version": "day7-v1",
        "corpus": "Day06: Bộ luật Lao động 2012, Bộ luật Lao động 2019, Nghị định 145/2020/NĐ-CP",
        "provenance": "200 câu có đáp án được chuyển từ benchmark Day06, giữ nguyên split và evidence; 4 câu thiếu evidence được khai báo riêng trong data/abstention_cases.json.",
        "split_note": "Day06 đã báo cáo kết quả trên 40 câu test; cấu hình Day06 được chọn bằng dev. Không dùng các câu test này để chọn cấu hình Day07. Nếu dùng kết quả test để điều chỉnh agent/retrieval, phải tạo test mới trước khi báo đánh giá độc lập.",
        "counts": {"dev_answerable": counts[("dev", True)], "dev_unanswerable": counts[("dev", False)],
                   "test_answerable": counts[("test", True)], "test_unanswerable": counts[("test", False)]},
        "questions": questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
