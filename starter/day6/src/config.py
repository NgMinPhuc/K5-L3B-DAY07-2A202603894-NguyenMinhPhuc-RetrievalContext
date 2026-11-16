"""Shared experiment settings for steps 4-5 (fixed before looking at results)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"

# Known output sizes; the embedder also checks the real vector length.
MODEL_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
# USD per 1M input tokens, OpenAI list price when written; re-check before reporting.
MODEL_PRICE_PER_1M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}

# Strategy -> which stored index it searches and whether question filters apply.
# C searches the same vectors as B; only the metadata filter differs (B-C comparison).
STRATEGIES = {
    "A": {"index": "A", "use_filter": False},
    "B": {"index": "B", "use_filter": False},
    "C": {"index": "B", "use_filter": True},
}
INDEXES = {"A": "chunks_A.jsonl", "B": "chunks_B.jsonl"}
TOP_K = 3

# Written 2026-11-14, before any benchmark run.
HYPOTHESES = {
    "A": "Baseline: chunk cố định 800/100 cắt ngang Điều/Khoản -> evidence có thể bị tách, Hit@3 thấp nhất.",
    "B": "Chunk theo Điều giữ trọn ý pháp lý -> Hit@3 cao hơn A; nhưng vẫn nhầm giữa BLLĐ 2012 và 2019 vì nội dung gần giống.",
    "C": "Như B + lọc year/document_id khi câu hỏi nêu văn bản -> tăng Hit@3 ở câu có filter, câu không filter giữ nguyên như B.",
}
# Pre-registered selection rule (applied to DEV only; TEST is reported, never used to choose).
SELECTION_RULE = "Chọn chiến lược có Hit@3 dev cao nhất; hòa thì Hit@1 dev; hòa tiếp thì grounding thủ công; cuối cùng là chi phí/thời gian."
# Manual audit samples, fixed before seeing results.
AUDIT_TEST_QUESTIONS = 5      # first 5 test questions by id (q016-q020; 2 of them carry a filter)
COHERENCE_POSITIONS = 10      # evenly spaced counted pages; anchor = text at the middle of the page
