"""Run the frozen baseline and selected config on the isolated TEST split.

The controller receives only the question and retrieved context. Gold labels
are used after each response for offline retrieval scoring and review output.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agentic import run_agent
from .citation_review import build_review_queue
from .dev_experiments import (
    CONFIGS, ROOT, _evaluate_trace, _make_config, _summarize, _usage_delta,
    _usage_snapshot,
)
from .model_controller import OpenAIController
from .retrieval import DenseRetriever, LexicalRetriever, load_day6_chunks
from .trace_store import TraceWriter, make_trace

BENCHMARK = ROOT / "data/test_benchmark_day7.json"
CHUNKS = ROOT / "starter/day6/data/chunks_B.jsonl"
CONFIG_NAMES = ("hybrid_rrf_baseline", "hybrid_rrf_citation_fix")


def run(output: Path, model: str | None = None, *, resume: bool = False) -> list[dict[str, Any]]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    questions = benchmark["questions"]
    if benchmark.get("split") != "test" or any(q.get("split") != "test" for q in questions):
        raise ValueError("Test export phải chỉ chứa câu test")
    if len(questions) != 42:
        raise ValueError(f"Kỳ vọng 42 câu test, nhận được {len(questions)}")

    started = time.perf_counter()
    chunks = load_day6_chunks(CHUNKS)
    lexical = LexicalRetriever(chunks)
    dense = DenseRetriever(index="B")
    setup_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    configs = [c for c in CONFIGS if c["name"] in CONFIG_NAMES]
    for config in configs:
        config_dir = output / config["name"]
        config_dir.mkdir(parents=True, exist_ok=True)
        trace_path = config_dir / "traces.jsonl"
        existing = []
        if resume and trace_path.exists():
            existing = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            existing = [trace for trace in existing if trace.get("status") != "error"]
            # Replace failed first attempts before retrying so the final trace
            # is a clean per-question record rather than mixed attempts.
            trace_path.write_text("".join(json.dumps(trace, ensure_ascii=False) + "\n"
                                               for trace in existing), encoding="utf-8")
        existing_by_id = {trace["query_id"]: trace for trace in existing}
        if any(qid not in {q["id"] for q in questions} for qid in existing_by_id):
            raise ValueError("Trace hiện có chứa ID ngoài test")
        if not resume:
            trace_path.write_text("", encoding="utf-8")
            existing_by_id = {}
        controller = OpenAIController(model=model)
        before = _usage_snapshot(controller)
        embed_before = getattr(dense.embedder, "total_tokens", 0) if dense.embedder else 0
        prior_chat_usage: dict[str, int | float] = {}
        for trace in existing:
            usage_row = trace.get("usage") or {}
            for key, value in (usage_row.get("chat_token_delta") or {}).items():
                if isinstance(value, (int, float)):
                    prior_chat_usage[key] = prior_chat_usage.get(key, 0) + value
        writer = TraceWriter(trace_path)
        for question in questions:
            if question["id"] in existing_by_id:
                continue
            case_before = _usage_snapshot(controller)
            case_embed_before = getattr(dense.embedder, "total_tokens", 0) if dense.embedder else 0
            start = time.perf_counter()
            error = None
            try:
                result = run_agent(question=question["question"], controller=controller,
                                   retriever=_make_config(config, dense, lexical),
                                   filters=question.get("filter"), max_searches=4,
                                   top_k=config["top_k"], budget=config["budget"])
            except Exception as exc:
                result = {"status": "error", "answer": None, "citations": [], "events": []}
                error = f"{type(exc).__name__}: {exc}"
            elapsed = (time.perf_counter() - start) * 1000
            usage = _usage_delta(case_before, _usage_snapshot(controller))
            embed_tokens = ((getattr(dense.embedder, "total_tokens", 0) if dense.embedder else 0)
                            - case_embed_before)
            trace = make_trace(query=question, result=result, config=config,
                               elapsed_ms=elapsed, cost_usd=None, error=error,
                               usage={"chat_token_delta": usage,
                                      "query_embedding_tokens": embed_tokens})
            trace["query_id"] = question["id"]
            trace["split"] = "test"
            writer.append(trace)
            existing_by_id[question["id"]] = trace

        traces = [existing_by_id[q["id"]] for q in questions if q["id"] in existing_by_id]
        rows = []
        by_question = {q["id"]: q for q in questions}
        for trace in traces:
            question = by_question[trace["query_id"]]
            result = {k: trace.get(k) for k in ("status", "answer", "citations", "events")}
            rows.append({**trace, "query_id": question["id"], "split": "test",
                         "answerable": bool(question["answerable"]), "question": question["question"],
                         "config": config["name"], "method": config["method"],
                         "top_k": config["top_k"], "context_budget_chars": config["budget"],
                         "gold_for_offline_scoring_only": question.get("gold", []),
                         "expected_answer_for_offline_review_only": question.get("expected_answer"),
                         "result": result, "evaluation": _evaluate_trace(question, result),
                         "elapsed_ms": trace.get("elapsed_ms", 0), "error": trace.get("error")})
        after = _usage_snapshot(controller)
        usage = _usage_delta(before, after)
        for key, value in prior_chat_usage.items():
            usage[key] = usage.get(key, 0) + value
        summary = _summarize(rows, config, setup_seconds,
                             sum((t.get("usage") or {}).get("query_embedding_tokens", 0)
                                 for t in traces), usage)
        summary["split"] = "test"
        summaries.append(summary)
        (config_dir / "scores.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (config_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        queue = build_review_queue(traces, questions=questions)
        (config_dir / "citation_review_queue.json").write_text(
            json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    aggregate = {
        "created_at": datetime.now(timezone.utc).isoformat(), "split": "test",
        "benchmark_path": str(BENCHMARK.relative_to(ROOT)), "test_questions": len(questions),
        "dev_used_for_test_selection": False, "model": model,
        "test_origin_note": "Câu answerable được kế thừa từ benchmark Day 6; Day 6 đã báo cáo test 40 câu. Day 7 dev đã dùng để chọn cấu hình, nhưng đây không phải bộ holdout hoàn toàn mới.",
        "configs": summaries,
    }
    (output / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy baseline và cấu hình đã chốt trên test")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/day7_test" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--model", help="Chat model; mặc định đọc OPENAI_CHAT_MODEL")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summaries = run(args.output, model=args.model, resume=args.resume)
    print(json.dumps({"output": str(args.output), "configs": [s["config"]["name"] for s in summaries]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
