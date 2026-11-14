"""Benchmark multiple retrieval methods (TF-IDF, BM25, pgvector Dense, Hybrid RRF) on dev/test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from .config import load_settings
from .embeddings import OpenAIQueryEmbedder
from .retrievers import (
    BM25Retriever,
    HybridRRFRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
    hit,
    pack_context,
)


def evaluate_method(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 2500,
    use_query_filter: bool = True,
    embedder: OpenAIQueryEmbedder | None = None,
) -> dict[str, Any]:
    tokens_before = embedder.total_prompt_tokens if embedder else 0
    cost_before = embedder.total_cost_usd if embedder else 0.0

    hit1 = 0
    hit3 = 0
    context_hit = 0
    context_hit_800 = 0
    context_chars: list[int] = []
    latencies_ms: list[float] = []
    traces: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for q in questions:
        filters = q.get("filter") if use_query_filter else None
        t0 = time.perf_counter()
        try:
            results = retriever.search(q["question"], top_k=top_k, filters=filters)
        except Exception as exc:
            errors.append({"question_id": q["id"], "error": str(exc)})
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        is_hit1 = bool(results and hit(results[0], q["gold"]))
        is_hit3 = any(hit(c, q["gold"]) for c in results[:3])

        ctx_text, ctx_ids = pack_context(results, budget=budget)
        by_id = {c["id"]: c for c in results}
        is_ctx_hit = any(hit(by_id[cid], q["gold"]) for cid in ctx_ids)

        _, ctx_ids_800 = pack_context(results, budget=800)
        is_ctx_hit_800 = any(hit(by_id[cid], q["gold"]) for cid in ctx_ids_800)

        hit1 += int(is_hit1)
        hit3 += int(is_hit3)
        context_hit += int(is_ctx_hit)
        context_hit_800 += int(is_ctx_hit_800)
        context_chars.append(len(ctx_text))

        traces.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "filter": filters,
                "hit1": is_hit1,
                "hit3": is_hit3,
                "context_hit": is_ctx_hit,
                "context_hit_800": is_ctx_hit_800,
                "context_ids": ctx_ids,
                "context_chars": len(ctx_text),
                "latency_ms": round(elapsed_ms, 3),
                "results": [
                    {
                        "id": r["id"],
                        "rank": r.get("rank"),
                        "score": r.get("score"),
                        "source_id": r["source_id"],
                        "page": r["page"],
                        "hit": hit(r, q["gold"]),
                        "dense_rank": r.get("dense_rank"),
                        "lexical_rank": r.get("lexical_rank"),
                    }
                    for r in results
                ],
            }
        )

    n = len(questions)
    tokens_used = (embedder.total_prompt_tokens - tokens_before) if embedder else 0
    cost_used = round((embedder.total_cost_usd - cost_before), 8) if embedder else 0.0

    summary = {
        "strategy": name,
        "n": n,
        "successful": len(traces),
        "errors": len(errors),
        "hit1": hit1,
        "hit1_fraction": f"{hit1}/{n}",
        "hit1_pct": round(100.0 * hit1 / n, 1) if n else 0.0,
        "hit3": hit3,
        "hit3_fraction": f"{hit3}/{n}",
        "hit3_pct": round(100.0 * hit3 / n, 1) if n else 0.0,
        "context_budget_chars": budget,
        "context_hit": context_hit,
        "context_hit_fraction": f"{context_hit}/{n}",
        "context_hit_pct": round(100.0 * context_hit / n, 1) if n else 0.0,
        "context_hit_800": context_hit_800,
        "context_hit_800_fraction": f"{context_hit_800}/{n}",
        "context_hit_800_pct": round(100.0 * context_hit_800 / n, 1) if n else 0.0,
        "avg_context_chars": round(sum(context_chars) / len(context_chars), 1) if context_chars else 0.0,
        "median_ms": round(statistics.median(latencies_ms), 3) if latencies_ms else 0.0,
        "total_ms": round(sum(latencies_ms), 3),
        "index_seconds": round(float(getattr(retriever, "index_seconds", 0.0)), 6),
        "embedding_tokens": tokens_used,
        "service_cost_usd": cost_used,
    }
    return {"summary": summary, "traces": traces, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods on Day 7 split.")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2500)
    parser.add_argument("--strategy", type=str, default=None, help="Day 6 pgvector strategy (default from .env)")
    parser.add_argument("--output", type=Path, default=Path("starter/results/dev-retrieval-benchmark"))
    args = parser.parse_args()

    settings = load_settings()
    day6_strategy = args.strategy or settings.day6_strategy

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q["split"] == args.split]
    if not questions:
        raise ValueError(f"No questions found for split={args.split!r}")

    embedder = OpenAIQueryEmbedder(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )
    tfidf = TfidfRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    dense = PgVectorDenseRetriever(
        database_url=settings.database_url,
        embedder=embedder,
        strategy=day6_strategy,
    )
    hybrid = HybridRRFRetriever(dense=dense, lexical=bm25, rrf_k=60, candidate_k=15)

    methods: list[tuple[str, Any, bool, OpenAIQueryEmbedder | None]] = [
        ("tfidf_no_filter", tfidf, False, None),
        ("bm25_filtered", bm25, True, None),
        ("dense_pgvector", dense, True, embedder),
        ("hybrid_rrf_dense_bm25", hybrid, True, embedder),
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for name, retriever, use_filter, emb in methods:
        res = evaluate_method(
            name,
            retriever,
            questions,
            top_k=args.top_k,
            budget=args.budget,
            use_query_filter=use_filter,
            embedder=emb,
        )
        summaries.append(res["summary"])
        (args.output / f"{name}_traces.json").write_text(
            json.dumps(res["traces"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if res["errors"]:
            (args.output / f"{name}_errors.json").write_text(
                json.dumps(res["errors"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    report = {
        "split": args.split,
        "corpus_chunks": len(chunks),
        "day6_strategy": day6_strategy,
        "embedding_model": settings.embedding_model,
        "top_k": args.top_k,
        "context_budget_chars": args.budget,
        "methods": summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
