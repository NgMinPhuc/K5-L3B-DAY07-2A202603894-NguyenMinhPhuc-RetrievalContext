"""Run Day 7 retrieval and agent experiments on DEV only.

From the repository root:
    python3 -m starter.dev_experiments

This script never sends expected answers or gold evidence to the controller.
It writes raw traces and review stubs; semantic answer/citation support still
requires human review against the original source.
"""
import argparse
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import unicodedata

from .agentic import run_agent
from .model_controller import OpenAIController
from .retrieval import DenseRetriever, HybridRetriever, LexicalRetriever, load_day6_chunks
from .trace_store import TraceWriter, make_trace
from .citation_review import build_review_queue


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/dev_benchmark_day7.json"
CHUNKS = ROOT / "starter/day6/data/chunks_B.jsonl"

CONFIGS = [
    {
        "name": "tfidf_baseline",
        "method": "tfidf",
        "top_k": 3,
        "budget": 800,
        "candidate_k": None,
        "hypothesis": "Lexical TF-IDF baseline; exact legal terms may help, but paraphrases may be missed.",
    },
    {
        "name": "dense_baseline",
        "method": "dense",
        "top_k": 3,
        "budget": 800,
        "candidate_k": None,
        "hypothesis": "Dense Day 6 index baseline; semantic similarity should retrieve paraphrases.",
    },
    {
        "name": "hybrid_rrf_baseline",
        "method": "hybrid_rrf",
        "top_k": 3,
        "budget": 800,
        "candidate_k": 20,
        "rrf_constant": 60,
        "hypothesis": "Fuse dense and lexical ranks equally with RRF to combine semantic and exact-term matches.",
    },
    {
        "name": "hybrid_rrf_wide_candidates",
        "method": "hybrid_rrf",
        "top_k": 3,
        "budget": 800,
        "candidate_k": 40,
        "rrf_constant": 60,
        "hypothesis": "Improvement 1: increase per-retriever candidates from 20 to 40; hypothesis: broader candidate recall gives RRF more chances to include evidence.",
    },
    {
        "name": "hybrid_rrf_more_context",
        "method": "hybrid_rrf",
        "top_k": 5,
        "budget": 1400,
        "candidate_k": 20,
        "rrf_constant": 60,
        "hypothesis": "Improvement 2: allow top-5 retrieval and 1400-character context; hypothesis: extra context helps when evidence is split or ranked just below top-3.",
    },
    {
        "name": "hybrid_rrf_citation_fix",
        "method": "hybrid_rrf",
        "top_k": 5,
        "budget": 1400,
        "candidate_k": 20,
        "rrf_constant": 60,
        "hypothesis": "Keep the selected dev retrieval settings; normalize model citations to plain IDs and ensure the answer includes bracketed citation markers to prevent structurally valid answers becoming abstentions.",
    },
]


def _squash(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).casefold().split())


def _is_hit(chunk: dict[str, Any], golds: list[dict[str, Any]]) -> bool:
    page_end = (chunk.get("metadata") or {}).get("page_end", chunk.get("page"))
    for gold in golds:
        if chunk.get("source_id") != gold["source_id"]:
            continue
        if not (chunk.get("page", 0) <= gold["page"] <= (page_end or chunk.get("page", 0))):
            continue
        if _squash(gold["evidence"]) in _squash(chunk.get("text", "")):
            return True
    return False


def _usage_snapshot(controller: Any) -> dict[str, Any]:
    """Read numeric telemetry only; tolerate controllers with no usage API."""
    values: dict[str, Any] = {}
    for name in ("total_prompt_tokens", "total_input_tokens", "prompt_tokens",
                 "total_completion_tokens", "total_output_tokens", "completion_tokens",
                 "total_tokens", "cost_usd"):
        value = getattr(controller, name, None)
        if isinstance(value, (int, float)):
            values[name] = value
    usage = getattr(controller, "usage", None)
    if isinstance(usage, dict):
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                values.setdefault(key, value)
    return values


def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {key: after[key] - before[key] for key in after.keys() & before.keys()
            if isinstance(after[key], (int, float))}


def _price_from_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} phải là số USD trên 1 triệu token") from exc
    if value < 0:
        raise ValueError(f"{name} không được âm")
    return value


def _chat_cost(usage: dict[str, Any]) -> tuple[float | None, str]:
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    input_price = _price_from_env("OPENAI_INPUT_PRICE_PER_1M_USD")
    output_price = _price_from_env("OPENAI_OUTPUT_PRICE_PER_1M_USD")
    if not isinstance(input_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
        return None, "Unavailable: controller did not report prompt/completion token deltas."
    if input_price is None or output_price is None:
        return None, "Unavailable: set OPENAI_INPUT_PRICE_PER_1M_USD and OPENAI_OUTPUT_PRICE_PER_1M_USD for the selected chat model."
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return cost, "Estimated from controller token deltas and configured per-million-token prices."


def _embedding_cost(embed_tokens: int) -> tuple[float | None, str]:
    explicit_price = _price_from_env("EMBEDDING_PRICE_PER_1M_USD")
    if explicit_price is not None:
        return embed_tokens * explicit_price / 1_000_000, "Configured EMBEDDING_PRICE_PER_1M_USD."
    try:
        from .day6.src.config import EMBEDDING_MODEL, MODEL_PRICE_PER_1M
        price = MODEL_PRICE_PER_1M.get(EMBEDDING_MODEL)
    except Exception:
        price = None
    if price is None:
        return None, "Unavailable: set EMBEDDING_PRICE_PER_1M_USD."
    return embed_tokens * price / 1_000_000, "Day 6 pinned price table; verify current provider pricing before reporting."


def _evaluate_trace(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    events = result.get("events", [])
    searches = [event for event in events if event.get("event") == "search"]
    first_results = searches[0].get("results", []) if searches else []
    all_results: dict[str, dict[str, Any]] = {}
    for event in searches:
        for chunk in event.get("results", []):
            all_results[chunk["id"]] = chunk
    golds = question.get("gold", []) if question.get("answerable") else []
    first_hit_ranks = [rank for rank, chunk in enumerate(first_results, 1) if _is_hit(chunk, golds)]
    final_context_ids = set(searches[-1].get("context_ids", [])) if searches else set()
    context_hit = any(chunk_id in final_context_ids and _is_hit(all_results[chunk_id], golds)
                      for chunk_id in all_results)
    retrieved_hit = any(_is_hit(chunk, golds) for chunk in all_results.values())
    citations = result.get("citations", [])
    citation_context_valid = bool(citations) and all(citation in final_context_ids for citation in citations)
    return {
        "first_hit1": bool(first_hit_ranks and first_hit_ranks[0] == 1),
        "first_evidence_hit3": any(rank <= 3 for rank in first_hit_ranks),
        "retrieved_evidence_any_turn": retrieved_hit,
        "context_evidence_present": context_hit if question.get("answerable") else None,
        "retrieval_calls": len(searches),
        "status": result.get("status", "error"),
        "abstention_correct": (result.get("status") == "abstain") if not question.get("answerable") else None,
        "answerable_abstained": (result.get("status") == "abstain") if question.get("answerable") else None,
        "citation_ids_in_final_context": citation_context_valid if citations else None,
        "citation_count": len(citations),
        "semantic_answer_grade": None,
        "semantic_citation_support": None,
        "manual_review_required": result.get("status") == "answered",
    }


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any], index_seconds: float,
               embed_tokens: int, controller_usage: dict[str, Any]) -> dict[str, Any]:
    labels = [row["evaluation"] for row in rows]
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    successful = [row for row in rows if row["evaluation"]["status"] != "error"]
    times = [row["elapsed_ms"] for row in rows]
    answer_grades = Counter(row["evaluation"]["semantic_answer_grade"] for row in rows
                            if row["evaluation"]["semantic_answer_grade"] is not None)
    citation_valid = [row["evaluation"]["citation_ids_in_final_context"] for row in rows
                      if row["evaluation"]["citation_ids_in_final_context"] is not None]
    abstentions = [row["evaluation"]["abstention_correct"] for row in unanswerable]
    answerable_abstentions = sum(bool(row["evaluation"]["answerable_abstained"]) for row in answerable)
    chat_cost, chat_cost_note = _chat_cost(controller_usage)
    embedding_cost, embedding_cost_note = _embedding_cost(embed_tokens)
    return {
        "config": config,
        "split": "dev",
        "n": len(rows),
        "errors": sum(row["evaluation"]["status"] == "error" for row in rows),
        "retrieval": {
            "hit1": {"n": sum(row["evaluation"]["first_hit1"] for row in answerable), "of": len(answerable)},
            "evidence_hit3": {"n": sum(row["evaluation"]["first_evidence_hit3"] for row in answerable), "of": len(answerable)},
            "retrieved_evidence_any_turn": {"n": sum(row["evaluation"]["retrieved_evidence_any_turn"] for row in answerable), "of": len(answerable)},
            "context_evidence_rate": {"n": sum(row["evaluation"]["context_evidence_present"] is True for row in answerable), "of": len(answerable)},
        },
        "agent": {
            "answered": sum(row["evaluation"]["status"] == "answered" for row in rows),
            "abstained": sum(row["evaluation"]["status"] == "abstain" for row in rows),
            "correct_abstentions": {"n": sum(abstentions), "of": len(unanswerable)},
            "answerable_abstentions": {"n": answerable_abstentions, "of": len(answerable)},
            "citation_ids_in_final_context": {"n": sum(citation_valid), "of": len(citation_valid)},
            "semantic_answer_grades": dict(answer_grades),
            "semantic_grading_note": "Answer correctness and semantic citation support are intentionally left for manual review; citation ID presence alone is not semantic validation.",
        },
        "latency": {
            "shared_index_build_seconds": round(index_seconds, 4),
            "mean_query_ms": round(statistics.mean(times), 2) if times else None,
            "median_query_ms": round(statistics.median(times), 2) if times else None,
            "successful_n": len(successful),
        },
        "usage": {
            "query_embedding_tokens": embed_tokens,
            "chat_usage": controller_usage,
            "embedding_estimated_cost_usd": embedding_cost,
            "embedding_cost_note": embedding_cost_note,
            "chat_estimated_cost_usd": chat_cost,
            "chat_cost_note": chat_cost_note,
            "total_estimated_cost_usd": (embedding_cost + chat_cost
                                         if embedding_cost is not None and chat_cost is not None else None),
        },
        "review_required": sum(row["evaluation"]["manual_review_required"] for row in rows),
    }


def _make_config(config: dict[str, Any], dense: DenseRetriever, lexical: LexicalRetriever) -> Any:
    if config["method"] == "dense":
        return dense
    if config["method"] == "tfidf":
        return lexical
    return HybridRetriever(dense, lexical,
                           candidate_k=config["candidate_k"],
                           rrf_constant=config.get("rrf_constant", 60))


def _row_from_trace(trace: dict[str, Any], question: dict[str, Any],
                    config: dict[str, Any]) -> dict[str, Any]:
    result = {"status": trace.get("status", "error"), "answer": trace.get("answer"),
              "citations": trace.get("citations", []), "events": trace.get("events", [])}
    error = ({"type": "recorded", "message": trace["error"]}
             if trace.get("error") else None)
    return {
        **trace,
        "query_id": question["id"], "split": "dev",
        "answerable": bool(question["answerable"]), "question": question["question"],
        "config": config["name"], "method": config["method"],
        "top_k": config["top_k"], "context_budget_chars": config["budget"],
        "gold_for_offline_scoring_only": question.get("gold", []) if question["answerable"] else [],
        "expected_answer_for_offline_review_only": question.get("expected_answer"),
        "result": result, "evaluation": _evaluate_trace(question, result),
        "elapsed_ms": trace.get("elapsed_ms", 0), "error": error,
    }


def run(output: Path, model: str | None = None, *, resume: bool = False,
        config_names: list[str] | None = None) -> list[dict[str, Any]]:
    # Read a physically separate DEV-only export. This file contains no test
    # question records, so they cannot enter controller, scoring, or selection.
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    dev_questions = benchmark["questions"]
    questions_by_id = {question["id"]: question for question in dev_questions}
    if any(q.get("split") != "dev" for q in dev_questions):
        raise ValueError("Dev-only benchmark có split khác dev")
    if not dev_questions:
        raise ValueError("Benchmark không có câu dev")
    chunk_load_started = time.perf_counter()
    chunks = load_day6_chunks(CHUNKS)
    chunk_load_seconds = time.perf_counter() - chunk_load_started
    index_started = time.perf_counter()
    lexical = LexicalRetriever(chunks)
    lexical_index_seconds = time.perf_counter() - index_started
    shared_setup_seconds = chunk_load_seconds + lexical_index_seconds
    dense = DenseRetriever(index="B")
    summaries = []
    output.mkdir(parents=True, exist_ok=True)
    available = {config["name"]: config for config in CONFIGS}
    selected_names = config_names or [config["name"] for config in CONFIGS]
    unknown = sorted(set(selected_names) - set(available))
    if unknown:
        raise ValueError(f"Tên cấu hình không hợp lệ: {', '.join(unknown)}")

    for name in selected_names:
        config = available[name]
        retriever = _make_config(config, dense, lexical)
        config_dir = output / config["name"]
        config_dir.mkdir(parents=True, exist_ok=True)
        trace_path = config_dir / "traces.jsonl"
        existing_traces = []
        if resume and trace_path.exists():
            existing_traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
                               if line.strip()]
            failed_traces = [trace for trace in existing_traces if trace.get("status") == "error"]
            if failed_traces:
                with (config_dir / "failed_attempts.jsonl").open("a", encoding="utf-8") as stream:
                    for trace in failed_traces:
                        stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
                existing_traces = [trace for trace in existing_traces if trace.get("status") != "error"]
                trace_path.write_text("".join(json.dumps(trace, ensure_ascii=False) + "\n"
                                                for trace in existing_traces), encoding="utf-8")
        existing_by_id = {trace["query_id"]: trace for trace in existing_traces}
        if resume and len(existing_by_id) == len(dev_questions) and (config_dir / "summary.json").exists():
            old_summary = json.loads((config_dir / "summary.json").read_text(encoding="utf-8"))
            if old_summary.get("errors", 0) == 0:
                summaries.append(old_summary)
                continue
        if not resume:
            trace_path.write_text("", encoding="utf-8")
            existing_traces = []
            existing_by_id = {}
        if any(query_id not in questions_by_id for query_id in existing_by_id):
            raise ValueError(f"Trace hiện có chứa ID ngoài dev trong {config['name']}")
        rows = [_row_from_trace(trace, questions_by_id[query_id], config)
                for query_id, trace in existing_by_id.items()]
        existing_chat_usage: dict[str, int | float] = {}
        existing_embed_tokens = 0
        for trace in existing_traces:
            usage = trace.get("usage") or {}
            existing_embed_tokens += int(usage.get("query_embedding_tokens", 0))
            for key, value in (usage.get("chat_token_delta") or {}).items():
                if isinstance(value, (int, float)):
                    existing_chat_usage[key] = existing_chat_usage.get(key, 0) + value
        # One controller per configuration keeps usage accounting separable.
        controller = OpenAIController(model=model)
        controller_before = _usage_snapshot(controller)
        embed_before = getattr(dense.embedder, "total_tokens", 0) if dense.embedder else 0
        trace_writer = TraceWriter(trace_path)
        # TF-IDF and chunk loading are shared prerequisites; the dense index was
        # built in Day 6. Per-query latency includes the query embedding/search.
        for question in dev_questions:
            if question["id"] in existing_by_id:
                continue
            start = time.perf_counter()
            case_usage_before = _usage_snapshot(controller)
            case_embed_before = getattr(dense.embedder, "total_tokens", 0) if dense.embedder else 0
            try:
                result = run_agent(
                    question=question["question"],
                    controller=controller,
                    retriever=retriever,
                    filters=question.get("filter"),
                    max_searches=4,
                    top_k=config["top_k"],
                    budget=config["budget"],
                )
                evaluation = _evaluate_trace(question, result)
                error = None
            except Exception as exc:
                result = {"status": "error", "answer": None, "citations": [], "events": []}
                evaluation = _evaluate_trace(question, result)
                error = {"type": type(exc).__name__, "message": str(exc)}
            elapsed_ms = (time.perf_counter() - start) * 1000
            trace_error = f"{error['type']}: {error['message']}" if error else None
            case_usage_delta = _usage_delta(case_usage_before, _usage_snapshot(controller))
            case_embed_tokens = ((getattr(dense.embedder, "total_tokens", 0) if dense.embedder else 0)
                                 - case_embed_before)
            case_chat_cost, case_chat_cost_note = _chat_cost(case_usage_delta)
            case_embedding_cost, case_embedding_cost_note = _embedding_cost(case_embed_tokens)
            case_total_cost = (case_chat_cost + case_embedding_cost
                               if case_chat_cost is not None and case_embedding_cost is not None else None)
            case_usage = {
                "chat_token_delta": case_usage_delta,
                "query_embedding_tokens": case_embed_tokens,
                "chat_cost_estimated_usd": case_chat_cost,
                "chat_cost_note": case_chat_cost_note,
                "embedding_cost_estimated_usd": case_embedding_cost,
                "embedding_cost_note": case_embedding_cost_note,
            }
            trace = make_trace(query=question, result=result, config=config,
                               elapsed_ms=elapsed_ms, cost_usd=case_total_cost,
                               error=trace_error, usage=case_usage)
            trace_writer.append(trace)
            rows.append({
                **trace,
                "query_id": question["id"],
                "split": "dev",
                "answerable": bool(question["answerable"]),
                "question": question["question"],
                "config": config["name"],
                "method": config["method"],
                "top_k": config["top_k"],
                "context_budget_chars": config["budget"],
                "gold_for_offline_scoring_only": question.get("gold", []) if question["answerable"] else [],
                "expected_answer_for_offline_review_only": question.get("expected_answer"),
                "result": result,
                "evaluation": evaluation,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": error,
            })

        controller_after = _usage_snapshot(controller)
        controller_delta = _usage_delta(controller_before, controller_after)
        for key, value in existing_chat_usage.items():
            controller_delta[key] = controller_delta.get(key, 0) + value
        embed_after = getattr(dense.embedder, "total_tokens", 0) if dense.embedder else 0
        summary = _summarize(rows, config, shared_setup_seconds,
                             existing_embed_tokens + embed_after - embed_before, controller_delta)
        summaries.append(summary)
        with (config_dir / "scores.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (config_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        review_queue = build_review_queue(trace_rows, questions=dev_questions)
        (config_dir / "citation_review_queue.json").write_text(
            json.dumps(review_queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "benchmark_path": str(BENCHMARK.relative_to(ROOT)),
        "dev_questions": len(dev_questions),
        "test_questions_file_read": False,
        "test_results_used_for_selection": False,
        "model": model,
        "configs": summaries,
        "selection_rule": "Chọn trên dev theo Evidence Hit@3 và Context Evidence Rate; nếu gần nhau, ưu tiên answer/citation sau khi review thủ công, rồi latency/cost. Không chọn dựa trên test.",
        "selection_status": "pending_manual_review",
    }
    (output / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy retrieval/agent experiments chỉ trên dev")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/day7_dev" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--model", help="Chat model; mặc định đọc OPENAI_CHAT_MODEL")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ trace JSONL đã ghi; không chạy lại câu đã hoàn tất")
    parser.add_argument("--configs", nargs="+", choices=[config["name"] for config in CONFIGS],
                        help="Chỉ chạy các cấu hình chỉ định; mặc định chạy toàn bộ")
    args = parser.parse_args()
    summaries = run(args.output, model=args.model, resume=args.resume, config_names=args.configs)
    print(json.dumps({"output": str(args.output), "configs": [s["config"]["name"] for s in summaries]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
