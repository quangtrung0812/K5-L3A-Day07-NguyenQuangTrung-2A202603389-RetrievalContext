"""Retrieval implementations for Day 7: TF-IDF, Okapi BM25, pgvector Dense, and Hybrid RRF."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math
import re
import time
import unicodedata
from typing import Any, Iterator, Protocol
from urllib.parse import urlparse

from .embeddings import QueryEmbedder

ALLOWED_FILTER_COLUMNS: dict[str, str] = {
    "domain": "domain",
    "year": "year",
    "version": "version",
    "section": "section",
    "document_type": "document_type",
    "document_id": "document_id",
}


class RetrieverProtocol(Protocol):
    name: str
    index_seconds: float

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    return any(
        chunk["source_id"] == g["source_id"]
        and chunk["page"] == g["page"]
        and g["evidence"] in chunk["text"]
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 2500) -> tuple[str, list[str]]:
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in results:
        if chunk["id"] in seen:
            continue
        seen.add(chunk["id"])
        block = f"[{chunk['id']} | {chunk['source']} | page {chunk['page']}]\n{chunk['text']}"
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(chunk["id"])
    return "\n\n".join(blocks), ids


def validate_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if not filters:
        return {}
    cleaned: dict[str, Any] = {}
    for key, val in filters.items():
        if val is None:
            continue
        if key not in ALLOWED_FILTER_COLUMNS:
            raise ValueError(f"Unsupported or disallowed filter key: {key!r}")
        cleaned[key] = val
    return cleaned


def matches_filter(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for key, val in filters.items():
        chunk_val = chunk.get("source_id") if key == "document_id" else meta.get(key)
        if chunk_val != val:
            return False
    return True


def _with_ranks(results: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for idx, item in enumerate(results, start=1):
        ranked.append({**item, "rank": idx, "strategy": strategy})
    return ranked


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


class TfidfRetriever:
    """In-memory TF-IDF lexical retriever with optional metadata filtering."""

    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        t0 = time.perf_counter()
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {
            term: math.log((1 + len(chunks)) / (1 + count)) + 1
            for term, count in df.items()
        }
        self.vectors = [self._vector(row) for row in counts]
        self.index_seconds = time.perf_counter() - t0

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vec = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        clean_filters = validate_filters(filters)

        qvec = self._vector(Counter(tokenize(query)))
        results: list[dict[str, Any]] = []
        for chunk, doc_vec in zip(self.chunks, self.vectors):
            if not matches_filter(chunk, clean_filters):
                continue
            score = sum(val * doc_vec.get(term, 0.0) for term, val in qvec.items())
            if score > 0:
                results.append({**chunk, "score": round(float(score), 6)})

        results.sort(key=lambda c: (-c["score"], c["id"]))
        return _with_ranks(results[:top_k], self.name)


class BM25Retriever:
    """Okapi BM25 lexical retriever with length normalization and metadata filtering."""

    name = "bm25"

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        t0 = time.perf_counter()
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_tfs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

        n_docs = len(chunks)
        df = Counter(term for tf in self.doc_tfs for term in tf)
        self.idf = {
            term: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }
        self.index_seconds = time.perf_counter() - t0

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        clean_filters = validate_filters(filters)

        q_terms = tokenize(query)
        results: list[dict[str, Any]] = []
        for chunk, tf_map, dl in zip(self.chunks, self.doc_tfs, self.doc_lens):
            if not matches_filter(chunk, clean_filters):
                continue
            score = 0.0
            norm_denom_base = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            for term in q_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * ((tf * (self.k1 + 1.0)) / (tf + norm_denom_base))
            if score > 0:
                results.append({**chunk, "score": round(float(score), 6)})

        results.sort(key=lambda c: (-c["score"], c["id"]))
        return _with_ranks(results[:top_k], self.name)


class PgVectorDenseRetriever:
    """Dense retriever querying Day 6 PostgreSQL + pgvector document_chunks table."""

    name = "dense"

    def __init__(
        self,
        database_url: str,
        embedder: QueryEmbedder,
        strategy: str = "B",
    ) -> None:
        if not database_url or not database_url.strip():
            raise RuntimeError("DATABASE_URL is missing; cannot connect to Day 6 pgvector store.")
        if not strategy or not strategy.strip():
            raise ValueError("strategy must not be empty")
        self.database_url = database_url.strip()
        self.embedder = embedder
        self.strategy = strategy.strip()
        self.index_seconds = 0.0  # Reuses pre-built Day 6 index

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        try:
            import psycopg

            conn = psycopg.connect(self.database_url, connect_timeout=10)
            try:
                yield conn
            finally:
                conn.close()
            return
        except ImportError:
            pass

        import pg8000.dbapi

        parsed = urlparse(self.database_url)
        conn = pg8000.dbapi.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 5432,
            database=(parsed.path or "/").lstrip("/"),
            timeout=10,
        )
        try:
            yield conn
        finally:
            conn.close()

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        clean_filters = validate_filters(filters)

        qvec = self.embedder.embed_query(query)
        qvec_lit = _vector_literal(qvec)

        where_clauses = ["strategy = %s"]
        params: list[Any] = [qvec_lit, self.strategy]
        for key, val in clean_filters.items():
            col = ALLOWED_FILTER_COLUMNS[key]
            where_clauses.append(f"{col} = %s")
            params.append(val)

        where_sql = " AND ".join(where_clauses)
        params.extend([qvec_lit, int(top_k)])

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                text,
                metadata,
                (embedding <=> %s::vector) AS cosine_distance
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s
        """

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for chunk_id, document_id, source, page, text, metadata, dist in rows:
            meta = json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {})
            score = 1.0 - float(dist)
            results.append(
                {
                    "id": chunk_id,
                    "source_id": document_id,
                    "source": source,
                    "page": int(page),
                    "text": text,
                    "metadata": meta,
                    "score": round(score, 6),
                }
            )
        return _with_ranks(results, self.name)


class HybridRRFRetriever:
    """Hybrid retriever fusing Dense (pgvector) and Lexical (BM25) via Reciprocal Rank Fusion."""

    name = "hybrid_rrf"

    def __init__(
        self,
        dense: RetrieverProtocol,
        lexical: RetrieverProtocol,
        rrf_k: int = 60,
        candidate_k: int = 15,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.index_seconds = getattr(lexical, "index_seconds", 0.0) + getattr(
            dense, "index_seconds", 0.0
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        clean_filters = validate_filters(filters)

        fetch_k = max(self.candidate_k, top_k)
        dense_hits = self.dense.search(query, top_k=fetch_k, filters=clean_filters)
        lex_hits = self.lexical.search(query, top_k=fetch_k, filters=clean_filters)

        fused: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        lex_ranks: dict[str, int] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            fused.setdefault(cid, item)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            dense_ranks[cid] = rank

        for rank, item in enumerate(lex_hits, start=1):
            cid = item["id"]
            fused.setdefault(cid, item)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            lex_ranks[cid] = rank

        merged: list[dict[str, Any]] = []
        for cid, item in fused.items():
            merged.append(
                {
                    **item,
                    "score": round(rrf_scores[cid], 6),
                    "dense_rank": dense_ranks.get(cid),
                    "lexical_rank": lex_ranks.get(cid),
                }
            )
        merged.sort(key=lambda c: (-c["score"], c["id"]))
        return _with_ranks(merged[:top_k], self.name)
