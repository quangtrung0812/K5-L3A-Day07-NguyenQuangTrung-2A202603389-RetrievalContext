"""Hybrid pgvector + BM25 retriever across text, tables, and visual index for WEF 2024."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import pg8000.dbapi

from ..embeddings import QueryEmbedder
from ..retrievers import BM25Retriever, _vector_literal, _with_ranks

DEFAULT_CHUNKS_PATH = Path("data/advanced/index/chunks.json")


class AdvancedHybridRetriever:
    """Searches 124-page WEF 2024 multimodal index (pgvector dense + BM25 + RRF)."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        database_url: str | None = None,
        embedder: QueryEmbedder | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.chunks_by_id = {c["id"]: c for c in chunks}
        self.database_url = (database_url or "").strip()
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(chunks)

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path = Path("data/advanced/index"),
        database_url: str | None = None,
        embedder: QueryEmbedder | None = None,
    ) -> "AdvancedHybridRetriever":
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        return cls(chunks=chunks, database_url=database_url, embedder=embedder)

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        parsed = urlparse(self.database_url)
        conn = pg8000.dbapi.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 5432,
            database=(parsed.path or "/").lstrip("/"),
            timeout=15,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _matches_scope(
        self,
        chunk: dict[str, Any],
        modality: str | None,
        page_start: int | None,
        page_end: int | None,
    ) -> bool:
        if modality and chunk.get("modality") != modality:
            return False
        pg = int(chunk.get("page", 0))
        if page_start is not None and pg < page_start:
            return False
        if page_end is not None and pg > page_end:
            return False
        return True

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        modality: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> list[dict[str, Any]]:
        raw = self.bm25.search(query, top_k=max(top_k * 6, 40), filters=None)
        filtered = [c for c in raw if self._matches_scope(c, modality, page_start, page_end)]
        return _with_ranks(filtered[:top_k], "bm25")

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        modality: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.database_url or not self.embedder:
            raise RuntimeError("Dense search requires database_url and embedder")
        qvec = self.embedder.embed_query(query)
        qvec_lit = _vector_literal(qvec)

        where_parts: list[str] = ["TRUE"]
        params: list[Any] = [qvec_lit]
        if modality:
            where_parts.append("modality = %s")
            params.append(modality)
        if page_start is not None:
            where_parts.append("page >= %s")
            params.append(int(page_start))
        if page_end is not None:
            where_parts.append("page <= %s")
            params.append(int(page_end))
        params.extend([qvec_lit, int(top_k)])

        sql = f"""
            SELECT
                chunk_id,
                source_id,
                page,
                modality,
                figure_id,
                bbox,
                crop_path,
                page_image_path,
                text,
                metadata,
                (embedding <=> %s::vector) AS cosine_distance
            FROM advanced_chunks
            WHERE {" AND ".join(where_parts)}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for (
            chunk_id,
            source_id,
            page,
            mod,
            figure_id,
            bbox,
            crop_path,
            page_image_path,
            text,
            metadata,
            dist,
        ) in rows:
            results.append(
                {
                    "id": chunk_id,
                    "source_id": source_id,
                    "source": "WEF Global Risks Report 2024",
                    "page": int(page),
                    "modality": mod,
                    "figure_id": figure_id,
                    "bbox": bbox if isinstance(bbox, list) else json.loads(bbox or "null"),
                    "crop_path": crop_path,
                    "page_image_path": page_image_path,
                    "text": text,
                    "metadata": metadata if isinstance(metadata, dict) else json.loads(metadata or "{}"),
                    "score": round(1.0 - float(dist), 6),
                }
            )
        return _with_ranks(results, "dense")

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        modality: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if mode == "bm25":
            return self.search_bm25(query, top_k=top_k, modality=modality, page_start=page_start, page_end=page_end)
        if mode == "dense":
            return self.search_dense(query, top_k=top_k, modality=modality, page_start=page_start, page_end=page_end)
        if mode != "hybrid":
            raise ValueError(f"Unsupported mode {mode!r}")

        cand_k = max(top_k * 4, 20)
        dense_res = self.search_dense(query, top_k=cand_k, modality=modality, page_start=page_start, page_end=page_end)
        bm25_res = self.search_bm25(query, top_k=cand_k, modality=modality, page_start=page_start, page_end=page_end)

        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for res_list in (dense_res, bm25_res):
            for rank, item in enumerate(res_list, 1):
                cid = item["id"]
                by_id[cid] = dict(item)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        fused = []
        for cid, rrf_score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]:
            item = dict(by_id[cid])
            item["score"] = round(rrf_score, 6)
            fused.append(item)
        return _with_ranks(fused, "hybrid_rrf")
