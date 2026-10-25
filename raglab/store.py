"""SQLite + FTS5 persistence and retrieval.

The platform deliberately starts with SQLite/FTS5 instead of a vector database:
it is deterministic, runs offline, and makes the retrieval failure modes easy to
inspect. The public surface is small enough to swap in another index later.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .chunking import chunk_text
from .tokenizer import token_text, tokenize


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    ordinal INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    body,
    tokenize='unicode61'
);
"""


class Store:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def add_document(self, title: str, text: str) -> Dict[str, Any]:
        title = (title or "").strip() or "未命名文档"
        text = (text or "").strip()
        if not text:
            raise ValueError("文档内容不能为空")
        if len(title) > 200:
            raise ValueError("标题不能超过 200 个字符")
        if len(text) > 200_000:
            raise ValueError("文档内容不能超过 200000 个字符")

        document_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        pieces = chunk_text(text)
        if not pieces:
            raise ValueError("文档没有可索引的内容")

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO documents (id, title, text, created_at) VALUES (?, ?, ?, ?)",
                (document_id, title, text, created_at),
            )
            for ordinal, piece in enumerate(pieces):
                chunk_id = "%s:%d" % (document_id, ordinal)
                connection.execute(
                    "INSERT INTO chunks (id, doc_id, title, text, ordinal) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, document_id, title, piece, ordinal),
                )
                connection.execute(
                    "INSERT INTO chunks_fts (chunk_id, body) VALUES (?, ?)",
                    (chunk_id, token_text(title + " " + piece)),
                )

        return {
            "id": document_id,
            "title": title,
            "chunks": len(pieces),
            "created_at": created_at,
        }

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        # Prefer multi-character terms for the relevance gate. Single Chinese
        # characters are useful for short queries but are noisy on long ones.
        signal_tokens = [token for token in tokens if len(token) > 1] or tokens
        signal_set = set(signal_tokens)
        match_query = " OR ".join('"%s"' % token for token in tokens)
        limit = max(top_k * 4, 12)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.doc_id, c.title, c.text, c.ordinal,
                       bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts) ASC
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()

        results: List[Dict[str, Any]] = []
        seen_per_doc: Dict[str, int] = {}
        for row in rows:
            doc_id = row["doc_id"]
            if seen_per_doc.get(doc_id, 0) >= 2:
                continue
            document_tokens = set(tokenize(row["title"] + " " + row["text"]))
            overlap = signal_set & document_tokens
            # Cap the denominator so long questions are not automatically
            # penalised for containing several cross-word bigrams.
            denominator = float(min(len(signal_set), 3))
            coverage = min(1.0, len(overlap) / denominator)
            if coverage <= 0.0:
                continue
            seen_per_doc[doc_id] = seen_per_doc.get(doc_id, 0) + 1
            results.append(
                {
                    "chunk_id": row["id"],
                    "doc_id": doc_id,
                    "title": row["title"],
                    "text": row["text"],
                    "ordinal": row["ordinal"],
                    "score": round(-float(row["rank"]), 4),
                    "coverage": round(coverage, 4),
                    "snippet": row["text"][:240],
                }
            )

        results.sort(key=lambda item: (-item["coverage"], -item["score"]))
        return results[:top_k]

    def count_documents(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def count_chunks(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.title, d.created_at, COUNT(c.id) AS chunks
                FROM documents d
                LEFT JOIN chunks c ON c.doc_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> Dict[str, int]:
        return {"documents": self.count_documents(), "chunks": self.count_chunks()}

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, doc_id, title, text, ordinal FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        return dict(row) if row else None
