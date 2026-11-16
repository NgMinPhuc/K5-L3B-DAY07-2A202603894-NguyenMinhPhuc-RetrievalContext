"""Store and search vectors in PostgreSQL + pgvector (exact cosine search, no ANN index)."""
import json
from typing import Any

import psycopg

from .models import Chunk

# Filter keys a question may use -> SQL column. Anything else is rejected.
FILTER_COLUMNS = {"year": "year", "document_id": "document_id",
                  "source_id": "source_id", "document_type": "document_type"}


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class PgVectorStore:
    TABLE = "chunks"

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("Thiếu DATABASE_URL")
        if dimensions <= 0:
            raise ValueError("dimensions phải > 0")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    strategy        text    NOT NULL,   -- stored index: A or B
                    chunk_id        text    NOT NULL,   -- stable: source:chunker:start-end
                    document_id     text    NOT NULL,
                    source_id       text    NOT NULL,
                    source          text    NOT NULL,
                    page            integer NOT NULL,
                    page_end        integer NOT NULL,
                    year            integer NOT NULL,
                    document_type   text    NOT NULL,
                    chapter         text,
                    article         text,
                    text            text    NOT NULL,
                    metadata        jsonb   NOT NULL,
                    content_hash    text    NOT NULL,   -- sha256(model + text)
                    embedding_model text    NOT NULL,
                    embedding       vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            row = conn.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'", (self.TABLE,)).fetchone()
            if row and row[0] != f"vector({self.dimensions})":
                raise RuntimeError(f"Bảng đã có cột {row[0]}, khác model hiện tại "
                                   f"vector({self.dimensions}). Đổi model thì phải tạo bảng mới.")

    def existing_hashes(self, strategy: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(f"SELECT chunk_id, content_hash FROM {self.TABLE} WHERE strategy = %s",
                                (strategy,)).fetchall()
        return dict(rows)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]],
               hashes: list[str], model_name: str) -> None:
        """Insert or update by (strategy, chunk_id): re-running never duplicates rows."""
        if not (len(chunks) == len(vectors) == len(hashes)):
            raise ValueError("Số chunk, vector và hash không khớp")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("Có vector sai số chiều")
        rows = []
        for c, v, h in zip(chunks, vectors, hashes):
            m = c.metadata
            rows.append((strategy, c.chunk_id, c.document_id, m["source_id"], c.source, c.page,
                         m["page_end"], m["year"], m["document_type"], m.get("chapter"),
                         m.get("article"), c.text, json.dumps(m, ensure_ascii=False), h,
                         model_name, _vec(v)))
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(f"""
                INSERT INTO {self.TABLE} (strategy, chunk_id, document_id, source_id, source, page,
                    page_end, year, document_type, chapter, article, text, metadata, content_hash,
                    embedding_model, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id, source_id = EXCLUDED.source_id,
                    source = EXCLUDED.source, page = EXCLUDED.page, page_end = EXCLUDED.page_end,
                    year = EXCLUDED.year, document_type = EXCLUDED.document_type,
                    chapter = EXCLUDED.chapter, article = EXCLUDED.article, text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model, embedding = EXCLUDED.embedding
                """, rows)

    def delete_obsolete(self, strategy: str, keep_ids: list[str]) -> int:
        """Remove chunks of this index that the current chunking no longer produces."""
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {self.TABLE} WHERE strategy = %s AND NOT chunk_id = ANY(%s)",
                               (strategy, keep_ids))
            return cur.rowcount

    def count(self, strategy: str) -> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT count(*) FROM {self.TABLE} WHERE strategy = %s",
                                (strategy,)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Exact cosine search; filters are whitelisted columns with bound parameters."""
        if len(query_vector) != self.dimensions:
            raise ValueError("Query vector sai số chiều")
        where, params = ["strategy = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f"Filter không được hỗ trợ: {key}")
            where.append(f"{FILTER_COLUMNS[key]} = %s")
            params.append(value)
        qv = _vec(query_vector)
        sql = (f"SELECT chunk_id, text, metadata, page, page_end, source_id, document_id, source, "
               f"1 - (embedding <=> %s::vector) AS score FROM {self.TABLE} "
               f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector LIMIT %s")
        with self._connect() as conn:
            rows = conn.execute(sql, [qv, *params, qv, top_k]).fetchall()
        return [{"chunk_id": r[0], "text": r[1], "metadata": r[2], "page": r[3], "page_end": r[4],
                 "source_id": r[5], "document_id": r[6], "source": r[7], "score": float(r[8])} for r in rows]
