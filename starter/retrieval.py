"""Dense, lexical TF-IDF, and RRF hybrid retrieval adapters for Day 7.

Run from the repository root and import as ``starter.retrieval``. Dense and
hybrid retrieval use the real Day 6 pgvector index; errors are surfaced.
"""
import json
from pathlib import Path
from typing import Any

from .baseline import TfidfIndex


def load_day6_chunks(path: Path) -> list[dict[str, Any]]:
    """Load Day 6 JSONL chunks and normalize them to the Day 7 chunk contract."""
    chunks = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            metadata = dict(raw.get("metadata") or {})
            metadata.setdefault("page_end", raw.get("page_end", raw.get("page")))
            chunks.append({
                "id": raw.get("id", raw.get("chunk_id")),
                "text": raw["text"],
                "source_id": raw.get("source_id", raw.get("document_id")),
                "source": raw["source"],
                "page": raw["page"],
                "metadata": metadata,
            })
            if not all(chunks[-1].get(key) is not None for key in ("id", "source_id", "source", "page")):
                raise ValueError(f"Chunk Day 6 thiếu trường bắt buộc ở dòng {line_no}")
    if not chunks:
        raise ValueError(f"Không có chunk trong {path}")
    return chunks


def _normalize_result(chunk: dict[str, Any], score: float, strategy: str) -> dict[str, Any]:
    return {
        "id": chunk.get("id", chunk.get("chunk_id")),
        "text": chunk["text"],
        "source_id": chunk["source_id"],
        "source": chunk["source"],
        "page": chunk["page"],
        "metadata": chunk.get("metadata") or {},
        "score": float(score),
        "strategy": strategy,
    }


class DenseRetriever:
    """Day 6 B dense index with the compatible OpenAI query embedder."""
    name = "dense"

    def __init__(self, index: str = "B") -> None:
        self.index = index
        self.embedder = None
        self.store = None

    def _connect(self) -> None:
        if self.embedder is not None:
            return
        from .day6.src.config import DATABASE_URL, EMBEDDING_MODEL
        from .day6.src.embeddings import OpenAIEmbedder
        from .day6.src.store import PgVectorStore

        if EMBEDDING_MODEL != "text-embedding-3-small":
            raise RuntimeError(
                "Index Day 6 dùng text-embedding-3-small (1536 chiều); "
                "đặt EMBEDDING_MODEL=text-embedding-3-small để tìm kiếm tương thích."
            )
        self.embedder = OpenAIEmbedder(model_name=EMBEDDING_MODEL)
        self.store = PgVectorStore(DATABASE_URL, self.embedder.dimensions)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        self._connect()
        vector = self.embedder.embed_query(query)
        hits = self.store.search(self.index, vector, top_k=top_k, filters=filters)
        return [_normalize_result(hit, hit["score"], self.name) for hit in hits]


class LexicalRetriever:
    """Offline TF-IDF retrieval over the copied Day 6 article chunks."""
    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        candidates = self.index.search(query, top_k=len(self.chunks))
        selected = []
        for hit in candidates:
            metadata = hit.get("metadata") or {}
            if filters and any(metadata.get(key) != value for key, value in filters.items()):
                continue
            selected.append(_normalize_result(hit, hit["score"], self.name))
            if len(selected) == top_k:
                break
        return selected


class HybridRetriever:
    """Fuse dense and TF-IDF ranks with Reciprocal Rank Fusion (RRF)."""
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, lexical: LexicalRetriever,
                 candidate_k: int = 20, rrf_constant: int = 60) -> None:
        if candidate_k < 1 or rrf_constant < 1:
            raise ValueError("candidate_k và rrf_constant phải >= 1")
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        ranked = [
            self.dense.search(query, filters=filters, top_k=self.candidate_k),
            self.lexical.search(query, filters=filters, top_k=self.candidate_k),
        ]
        fused: dict[str, dict[str, Any]] = {}
        for results in ranked:
            for rank, hit in enumerate(results, 1):
                row = fused.setdefault(hit["id"], {**hit, "score": 0.0, "rrf_ranks": {}})
                row["score"] += 1.0 / (self.rrf_constant + rank)
                row["rrf_ranks"][hit["strategy"]] = rank
        output = sorted(fused.values(), key=lambda hit: (-hit["score"], hit["id"]))[:top_k]
        for hit in output:
            hit["strategy"] = self.name
        return output


def build_retrievers(chunks_path: Path | None = None) -> dict[str, Any]:
    """Build all three methods; API/database setup is lazy until dense search."""
    if chunks_path is None:
        chunks_path = Path(__file__).resolve().parent / "day6/data/chunks_B.jsonl"
    lexical = LexicalRetriever(load_day6_chunks(chunks_path))
    dense = DenseRetriever(index="B")
    return {
        "dense": dense,
        "tfidf": lexical,
        "hybrid_rrf": HybridRetriever(dense, lexical, candidate_k=20, rrf_constant=60),
    }
