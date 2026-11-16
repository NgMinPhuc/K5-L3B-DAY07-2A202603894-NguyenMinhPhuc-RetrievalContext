"""Create a human citation-audit queue from agent traces.

The script checks structural membership only. A citation ID in context does
not show that its text supports the answer; that decision must be reviewed.
"""
import argparse
import json
from pathlib import Path
from typing import Any


def build_review_queue(traces: list[dict[str, Any]],
                       questions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    question_map = {question["id"]: question for question in (questions or [])}
    queue = []
    for trace in traces:
        events = trace.get("events") or []
        searches = [event for event in events if event.get("event") == "search"]
        final_ids = set(searches[-1].get("context_ids", [])) if searches else set()
        chunks: dict[str, dict[str, Any]] = {}
        for event in searches:
            for hit in event.get("results", []):
                if hit.get("id"):
                    chunks[hit["id"]] = hit
        citations = trace.get("citations") or []
        references = []
        for citation in citations:
            hit = chunks.get(citation)
            references.append({
                "citation_id": citation,
                "in_final_context": citation in final_ids,
                "source": hit.get("source") if hit else None,
                "source_id": hit.get("source_id") if hit else None,
                "page": hit.get("page") if hit else None,
                "text_to_review": hit.get("text") if hit else None,
                "supports_answer": None,
            })
        source_question = question_map.get(trace["query_id"], {})
        queue.append({
            "query_id": trace["query_id"],
            "split": trace.get("split"),
            "question": trace.get("question"),
            "status": trace.get("status"),
            "answer": trace.get("answer"),
            "expected_answer_for_offline_review_only": source_question.get("expected_answer"),
            "gold_for_offline_review_only": source_question.get("gold", []),
            "citation_references": references,
            "citation_context_membership": [ref["in_final_context"] for ref in references],
            "citation_support": [None for _ in references],
            "answer_grade": None,
            "abstention_correct": None,
            "review_note": "Người chấm: đối chiếu text/page với nguồn gốc; ID có trong context không tự chứng minh citation hỗ trợ câu trả lời.",
        })
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo hàng đợi audit citation để người chấm điền")
    parser.add_argument("--traces", type=Path, required=True, help="JSON array hoặc JSONL trace")
    parser.add_argument("--output", type=Path, required=True, help="JSON review template")
    parser.add_argument("--benchmark", type=Path,
                        help="Optional benchmark JSON with questions; included only in offline review output")
    args = parser.parse_args()
    raw = args.traces.read_text(encoding="utf-8")
    if args.traces.suffix.lower() == ".jsonl":
        traces = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        traces = json.loads(raw)
    if not isinstance(traces, list):
        raise ValueError("Trace phải là JSON array hoặc JSONL")
    questions = None
    if args.benchmark:
        benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
        questions = benchmark.get("questions", benchmark) if isinstance(benchmark, dict) else benchmark
        if not isinstance(questions, list):
            raise ValueError("Benchmark phải chứa danh sách questions")
    reviews = build_review_queue(traces, questions=questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Tạo {len(reviews)} review template tại {args.output}; các nhãn hỗ trợ vẫn để trống chờ người chấm.")


if __name__ == "__main__":
    main()
