"""Run advanced-challenge configs on a benchmark split and save per-question traces.

The agent only sees the question; gold pages/answers are used afterwards for
offline retrieval metrics and for the pass/fail review queue.

  .venv/bin/python -m advanced.run_eval --split dev --configs text_only multimodal ...
  .venv/bin/python -m advanced.run_eval --split test --configs <frozen config>
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from starter.agentic import run_agent

from . import rag

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "advanced/benchmark"

CONFIGS: dict[str, dict[str, Any]] = {
    "text_only": {"method": "hybrid_rrf", "modalities": ["text"], "top_k": 6, "budget": 6000,
                  "page_images": False,
                  "hypothesis": "Baseline: page text layer only; tables partly survive as text, charts/figures do not."},
    "multimodal": {"method": "hybrid_rrf", "modalities": None, "top_k": 6, "budget": 6000, "page_images": False,
                   "hypothesis": "Adding vision transcriptions of tables/charts/figures lets retrieval reach visual evidence."},
    "multimodal_dense": {"method": "dense", "modalities": None, "top_k": 6, "budget": 6000, "page_images": False,
                         "hypothesis": "Dense-only over the same chunks, to compare against RRF fusion."},
    "multimodal_wide": {"method": "hybrid_rrf", "modalities": None, "top_k": 8, "budget": 10000, "page_images": False,
                        "hypothesis": "More context helps multi-hop and long tables."},
    "multimodal_pageimg": {"method": "hybrid_rrf", "modalities": None, "top_k": 6, "budget": 6000, "page_images": True,
                           "hypothesis": "Attaching rendered context pages lets the model read charts/tables directly, fixing transcription errors."},
    "multimodal_wide_pageimg": {"method": "hybrid_rrf", "modalities": None, "top_k": 8, "budget": 10000,
                                "page_images": True, "hypothesis": "Combine wide context and page images."},
    "v2_wide": {"method": "hybrid_rrf", "modalities": None, "top_k": 8, "budget": 10000, "page_images": False,
                "label_boost": True, "prompt": "v2", "max_searches": 6,
                "hypothesis": "Index v2 (annex/section headers, continued-table labels) + label boost + prompt v2 (verify facts in question, one search per part, strict period matching)."},
    "v2_wide_pageimg": {"method": "hybrid_rrf", "modalities": None, "top_k": 8, "budget": 10000, "page_images": True,
                        "label_boost": True, "prompt": "v2", "max_searches": 6,
                        "hypothesis": "v2_wide plus rendered context pages for reading charts/tables directly."},
}


def _price(name: str) -> float | None:
    v = os.environ.get(name)
    return float(v) if v else None


def _pages_of(ids: list[str]) -> set[int]:
    return {int(m.group(1)) for i in ids if (m := re.match(r"p(\d+)-", i))}


def _evaluate(q: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    gold_pages = {int(e["page"]) for e in q.get("evidence", [])}
    searches = [e for e in result.get("events", []) if e.get("event") == "search"]
    retrieved = _pages_of([r["id"] for e in searches for r in e["results"]])
    final_ctx = _pages_of(searches[-1]["context_ids"]) if searches else set()
    cited = _pages_of([c["id"] if isinstance(c, dict) else c for c in result.get("citations") or []])
    return {
        "gold_pages": sorted(gold_pages),
        "retrieved_pages": sorted(retrieved), "context_pages": sorted(final_ctx), "cited_pages": sorted(cited),
        "all_gold_pages_retrieved": bool(gold_pages) and gold_pages <= retrieved,
        "all_gold_pages_in_context": bool(gold_pages) and gold_pages <= final_ctx,
        "all_gold_pages_cited": bool(gold_pages) and gold_pages <= cited,
        "searches": len(searches),
    }


def run(split: str, names: list[str], out: Path, *, workers: int = 6) -> dict[str, Any]:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    bench = json.loads((BENCH / f"{split}.json").read_text(encoding="utf-8"))
    questions = bench["questions"]
    if any(q.get("split", split) != split for q in questions):
        raise ValueError("benchmark chứa câu khác split")
    chunks = rag.load_chunks()
    by_id = {c["id"]: c for c in chunks}
    dense, lexical = rag.LocalDense(chunks), rag.FilteredLexical(chunks)
    pin, pout = _price("OPENAI_INPUT_PRICE_PER_1M_USD"), _price("OPENAI_OUTPUT_PRICE_PER_1M_USD")
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in names:
        cfg = CONFIGS[name]
        cdir = out / name
        cdir.mkdir(parents=True, exist_ok=True)
        retriever = rag.build(chunks, cfg["method"], modalities=cfg["modalities"], dense=dense, lexical=lexical)
        if cfg.get("label_boost"):
            retriever = rag.LabelBoost(retriever, chunks)
        prompt = rag.SYSTEM_PROMPT_V2 if cfg.get("prompt") == "v2" else rag.SYSTEM_PROMPT_EN

        def one(q: dict[str, Any]) -> dict[str, Any]:
            ctrl_cls = rag.PageImageController if cfg["page_images"] else rag.OpenAIController
            ctrl = ctrl_cls(system_prompt=prompt, max_completion_tokens=1500)
            t0 = time.perf_counter()
            err = None
            try:
                res = run_agent(q["question"], ctrl, retriever, max_searches=cfg.get("max_searches", 5),
                                top_k=cfg["top_k"], budget=cfg["budget"])
            except Exception as exc:
                res = {"status": "error", "answer": None, "citations": [], "events": []}
                err = f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t0) * 1000
            u = ctrl.usage
            cost = (u["prompt_tokens"] * pin + u["completion_tokens"] * pout) / 1e6 if pin and pout else None
            cites = [{"id": c, "page": by_id[c]["page"], "modality": by_id[c]["metadata"].get("modality"),
                      "visual_type": by_id[c]["metadata"].get("visual_type"),
                      "label": by_id[c]["metadata"].get("label"), "region": by_id[c]["metadata"].get("region"),
                      "caption_bbox": by_id[c]["metadata"].get("caption_bbox"), "text": by_id[c]["text"]}
                     for c in res.get("citations") or [] if c in by_id]
            return {"query_id": q["id"], "split": split, "group": q["group"], "config": name,
                    "question": q["question"], "status": res["status"], "answer": res.get("answer"),
                    "citations": cites, "events": res.get("events", []), "error": err,
                    "elapsed_ms": round(ms, 1), "usage": dict(u),
                    "page_images_sent": getattr(ctrl, "images_sent", 0), "chat_cost_usd": cost}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            traces = list(pool.map(one, questions))
        with (cdir / "traces.jsonl").open("w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        qmap = {q["id"]: q for q in questions}
        rows, queue = [], []
        for t in traces:
            q = qmap[t["query_id"]]
            ev = _evaluate(q, t)
            rows.append({"query_id": t["query_id"], "group": q["group"], "answerable": q["answerable"],
                         "status": t["status"], "error": t["error"], "elapsed_ms": t["elapsed_ms"], **ev})
            queue.append({"query_id": t["query_id"], "group": q["group"], "question": q["question"],
                          "status": t["status"], "answer": t["answer"],
                          "citations": [{k: c[k] for k in ("id", "page", "modality", "visual_type", "label", "region", "caption_bbox", "text")}
                                        for c in t["citations"]],
                          "expected_for_offline_review_only": q.get("expected_answer"),
                          "gold_evidence_for_offline_review_only": q.get("evidence"),
                          "pass": None, "review_reason": None})
        (cdir / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (cdir / "review_queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ans = [r for r in rows if r["answerable"]]
        lat = sorted(t["elapsed_ms"] for t in traces)
        s = {"config": {"name": name, **cfg}, "split": split, "n": len(rows),
             "errors": sum(bool(r["error"]) for r in rows),
             "answered": sum(r["status"] == "answered" for r in rows),
             "abstained_unanswerable": sum(r["status"] == "abstain" and not r["answerable"] for r in rows),
             "unanswerable": sum(not r["answerable"] for r in rows),
             "all_gold_pages_retrieved": f"{sum(r['all_gold_pages_retrieved'] for r in ans)}/{len(ans)}",
             "all_gold_pages_in_context": f"{sum(r['all_gold_pages_in_context'] for r in ans)}/{len(ans)}",
             "all_gold_pages_cited": f"{sum(r['all_gold_pages_cited'] for r in ans)}/{len(ans)}",
             "mean_ms": round(sum(lat) / len(lat), 1), "median_ms": lat[len(lat) // 2],
             "chat_tokens": {k: sum(t["usage"][k] for t in traces) for k in ("prompt_tokens", "completion_tokens")},
             "chat_cost_usd": round(sum(t["chat_cost_usd"] or 0 for t in traces), 4) if pin and pout else None,
             "pass": None, "pass_note": "Fill after reviewing review_queue.json against the original PDF."}
        (cdir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(s)
        print(json.dumps({k: v for k, v in s.items() if k != "config"}, ensure_ascii=False), flush=True)
    agg = {"created_at": datetime.now(timezone.utc).isoformat(), "split": split, "configs": summaries}
    (out / "summary.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    run(a.split, a.configs, a.output, workers=a.workers)


if __name__ == "__main__":
    main()
