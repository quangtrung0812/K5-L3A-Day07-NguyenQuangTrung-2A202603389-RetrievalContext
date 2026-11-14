"""Run locked baseline (cfg0) and selected (cfg4) configurations on the holdout test split (7 cases)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import AgentConfig
from .benchmark_retrieval import evaluate_method
from .config import load_settings
from .embeddings import OpenAIQueryEmbedder
from .retrievers import (
    BM25Retriever,
    HybridRRFRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
)
from .run_agent_batch import run_batch


def main() -> None:
    settings = load_settings()
    chunks = json.loads(Path("starter/data/converted/chunks.json").read_text(encoding="utf-8"))
    all_questions = json.loads(Path("starter/data/converted/questions.json").read_text(encoding="utf-8"))
    test_questions = [q for q in all_questions if q["split"] == "test"]
    all_cases = json.loads(Path("starter/data/converted/agent_cases.json").read_text(encoding="utf-8"))
    test_cases = [c for c in all_cases if c["split"] == "test"]

    # 1. Retrieval-only benchmark on test (5 answerable questions)
    embedder = OpenAIQueryEmbedder(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )
    tfidf = TfidfRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    dense = PgVectorDenseRetriever(
        database_url=settings.database_url,
        embedder=embedder,
        strategy=settings.day6_strategy,
    )
    hybrid = HybridRRFRetriever(dense=dense, lexical=bm25, rrf_k=60, candidate_k=15)

    ret_out = Path("starter/results/test-retrieval-benchmark")
    ret_out.mkdir(parents=True, exist_ok=True)
    ret_summaries: list[dict[str, Any]] = []
    for name, retriever, use_filter, emb in (
        ("tfidf_no_filter", tfidf, False, None),
        ("bm25_filtered", bm25, True, None),
        ("dense_pgvector", dense, True, embedder),
        ("hybrid_rrf_dense_bm25", hybrid, True, embedder),
    ):
        res = evaluate_method(
            name,
            retriever,
            test_questions,
            top_k=3,
            budget=2500,
            use_query_filter=use_filter,
            embedder=emb,
        )
        ret_summaries.append(res["summary"])
        (ret_out / f"{name}_traces.json").write_text(
            json.dumps(res["traces"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    ret_report = {
        "split": "test",
        "corpus_chunks": len(chunks),
        "day6_strategy": settings.day6_strategy,
        "embedding_model": settings.embedding_model,
        "top_k": 3,
        "context_budget_chars": 2500,
        "methods": ret_summaries,
    }
    (ret_out / "summary.json").write_text(
        json.dumps(ret_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 2. Agent baseline (cfg0) vs locked selected (cfg4) on all 7 test cases
    configs: list[tuple[str, Path, AgentConfig]] = [
        (
            "cfg0_baseline_dense_800",
            Path("starter/results/test-agent-baseline"),
            AgentConfig(
                model=settings.agent_model,
                default_mode="dense",
                max_searches=2,
                top_k=3,
                context_budget=800,
                context_mode="first_come",
                allow_agent_filter=False,
            ),
        ),
        (
            "cfg4_hybrid_4000_k5_followup",
            Path("starter/results/test-agent-selected"),
            AgentConfig(
                model=settings.agent_model,
                default_mode="hybrid",
                max_searches=4,
                top_k=5,
                context_budget=4000,
                context_mode="score_ranked",
                allow_agent_filter=True,
            ),
        ),
    ]

    agent_summaries: list[dict[str, Any]] = []
    for name, out_dir, cfg in configs:
        res = run_batch(
            test_cases,
            chunks,
            cfg,
            day6_strategy=settings.day6_strategy,
            database_url=settings.database_url,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.embedding_model,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "traces.json").write_text(
            json.dumps(res["traces"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary = {"experiment": name, **res["summary"]}
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        agent_summaries.append(summary)

    print(json.dumps({"retrieval_test": ret_summaries, "agent_test": agent_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
