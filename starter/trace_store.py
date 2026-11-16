"""Append and normalize complete per-query Day 7 agent traces as JSONL."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_trace(*, query: dict[str, Any], result: dict[str, Any], config: dict[str, Any],
               elapsed_ms: float, cost_usd: float | None = None,
               error: str | None = None,
               usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create one durable trace record; do not include gold/expected answers."""
    if not query.get("id") or query.get("split") not in ("dev", "test"):
        raise ValueError("query cần id và split dev/test")
    events = result.get("events", [])
    if not isinstance(events, list):
        raise ValueError("result.events phải là list")
    status = "error" if error else result.get("status", "error")
    if status not in ("answered", "abstain", "error"):
        raise ValueError(f"status không hợp lệ: {status}")
    return {
        "query_id": query["id"],
        "split": query["split"],
        "question": query["question"],
        "status": status,
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "events": events,
        "config": dict(config),
        "elapsed_ms": round(float(elapsed_ms), 3),
        "cost_usd": cost_usd,
        "usage": usage or {},
        "error": error,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


class TraceWriter:
    """Append one JSON object per line; flush after every case for resilience."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: dict[str, Any]) -> None:
        if not trace.get("query_id"):
            raise ValueError("trace thiếu query_id")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL dòng {line_no} không phải object")
                rows.append(row)
    return rows
