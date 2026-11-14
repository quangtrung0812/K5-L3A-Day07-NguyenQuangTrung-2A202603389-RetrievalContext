"""Unit tests for Day 7 retrieval implementations (BM25, TF-IDF, Hybrid RRF, error handling)."""
from __future__ import annotations

import unittest

from src.retrievers import (
    BM25Retriever,
    HybridRRFRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
    matches_filter,
    pack_context,
    validate_filters,
)

SAMPLE_CHUNKS = [
    {
        "id": "doc1:p1:c1",
        "source_id": "doc1",
        "source": "doc1.pdf",
        "page": 1,
        "text": "Trustworthy AI systems are valid, reliable, safe, secure, accountable, and fair.",
        "metadata": {"domain": "ai_risk", "year": 2023, "section": "sec_1"},
    },
    {
        "id": "doc1:p2:c1",
        "source_id": "doc1",
        "source": "doc1.pdf",
        "page": 2,
        "text": "Third-party AI governance requires contingency plans for external supplier failures.",
        "metadata": {"domain": "ai_risk", "year": 2023, "section": "govern"},
    },
    {
        "id": "doc2:p1:c1",
        "source_id": "doc2",
        "source": "doc2.pdf",
        "page": 1,
        "text": "Cybersecurity Framework 2.0 introduces the Govern function alongside Identify, Protect, Detect, Respond, and Recover.",
        "metadata": {"domain": "cybersecurity", "year": 2024, "section": "core"},
    },
]


class _FakeDense:
    name = "dense"
    index_seconds = 0.0

    def search(self, query, top_k=3, filters=None):
        clean = validate_filters(filters)
        candidates = [c for c in SAMPLE_CHUNKS if matches_filter(c, clean)]
        # Favor doc1:p2:c1 first, then doc2:p1:c1
        ordered = sorted(candidates, key=lambda c: 0 if c["id"] == "doc1:p2:c1" else 1)
        return [{**c, "score": 0.9 - i * 0.1, "rank": i + 1, "strategy": "dense"} for i, c in enumerate(ordered[:top_k])]


class _DummyEmbedder:
    model_name = "text-embedding-3-small"
    dimensions = 1536

    def embed_query(self, query: str) -> list[float]:
        return [0.0] * 1536


class RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_matching_chunk_and_applies_filter(self):
        bm25 = BM25Retriever(SAMPLE_CHUNKS)
        hits = bm25.search("contingency plans third-party supplier", top_k=3)
        self.assertEqual(hits[0]["id"], "doc1:p2:c1")
        self.assertEqual(hits[0]["strategy"], "bm25")
        self.assertEqual(hits[0]["rank"], 1)

        filtered = bm25.search("Govern", top_k=3, filters={"domain": "cybersecurity", "year": 2024})
        self.assertEqual([h["id"] for h in filtered], ["doc2:p1:c1"])

    def test_tfidf_supports_filters(self):
        tfidf = TfidfRetriever(SAMPLE_CHUNKS)
        filtered = tfidf.search("trustworthy AI", top_k=3, filters={"year": 2023})
        self.assertTrue(all(h["metadata"]["year"] == 2023 for h in filtered))

    def test_hybrid_rrf_fuses_and_deduplicates(self):
        bm25 = BM25Retriever(SAMPLE_CHUNKS)
        hybrid = HybridRRFRetriever(dense=_FakeDense(), lexical=bm25, rrf_k=60, candidate_k=5)
        hits = hybrid.search("contingency plans third-party", top_k=3)
        self.assertEqual(hits[0]["id"], "doc1:p2:c1")
        self.assertEqual(hits[0]["strategy"], "hybrid_rrf")
        self.assertEqual(len({h["id"] for h in hits}), len(hits))
        self.assertEqual(hits[0]["dense_rank"], 1)
        self.assertEqual(hits[0]["lexical_rank"], 1)

    def test_error_handling_on_invalid_inputs_and_missing_config(self):
        bm25 = BM25Retriever(SAMPLE_CHUNKS)
        with self.assertRaises(ValueError):
            bm25.search("   ")
        with self.assertRaises(ValueError):
            bm25.search("AI", top_k=0)
        with self.assertRaises(ValueError):
            bm25.search("AI", filters={"unknown_column": "x"})
        with self.assertRaises(RuntimeError):
            PgVectorDenseRetriever(database_url="", embedder=_DummyEmbedder())

    def test_pack_context_respects_budget_and_dedup(self):
        dup_results = [SAMPLE_CHUNKS[0], SAMPLE_CHUNKS[0], SAMPLE_CHUNKS[1]]
        ctx, ids = pack_context(dup_results, budget=300)
        self.assertEqual(ids, ["doc1:p1:c1", "doc1:p2:c1"])
        self.assertLessEqual(len(ctx), 300)


if __name__ == "__main__":
    unittest.main()
