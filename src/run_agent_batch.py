"""Batch runner for real Agentic RAG evaluation on Day 7 dev/test cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from .agent import AgentConfig, MultiToolRetriever, OpenAIChatClient, run_agentic_rag
from .config import load_settings
from .embeddings import OpenAIQueryEmbedder
from .retrievers import BM25Retriever, HybridRRFRetriever, PgVectorDenseRetriever, hit


def run_batch(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    config: AgentConfig,
    *,
    day6_strategy: str,
    database_url: str,
    openai_api_key: str,
    embedding_model: str,
) -> dict[str, Any]:
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing in .env")

    client = OpenAIChatClient(api_key=openai_api_key, timeout=60.0)
    embedder = OpenAIQueryEmbedder(api_key=openai_api_key, model_name=embedding_model)
    dense = PgVectorDenseRetriever(
        database_url=database_url,
        embedder=embedder,
        strategy=day6_strategy,
    )
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRRFRetriever(dense=dense, lexical=bm25, rrf_k=60, candidate_k=15)
    multi = MultiToolRetriever(dense=dense, bm25=bm25, hybrid=hybrid)

    chunk_by_id = {c["id"]: c for c in chunks}
    traces: list[dict[str, Any]] = []

    for case in cases:
        qid = case["id"]
        try:
            trace = run_agentic_rag(
                case["question"],
                client=client,
                retriever=multi,
                config=config,
                user_filters=case.get("filter"),
                embedder=embedder,
                query_id=qid,
            )
        except Exception as exc:
            trace = {
                "query_id": qid,
                "question": case["question"],
                "status": "error",
                "answer": "",
                "citations": [],
                "search_calls": 0,
                "events": [],
                "final_context_ids": [],
                "final_context": "",
                "model": config.model,
                "llm_calls": 0,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "embedding_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        traces.append(trace)

    ok_traces = [t for t in traces if t["status"] != "error"]
    err_traces = [t for t in traces if t["status"] == "error"]
    case_by_id = {c["id"]: c for c in cases}

    # Post-run context evidence check (gold is never passed into agent).
    ans_ok = [t for t in ok_traces if case_by_id[t["query_id"]]["answerable"]]
    ctx_hits = sum(
        1
        for t in ans_ok
        if any(
            cid in chunk_by_id and hit(chunk_by_id[cid], case_by_id[t["query_id"]]["gold"])
            for cid in t.get("final_context_ids", [])
        )
    )
    unans_ok = [t for t in ok_traces if not case_by_id[t["query_id"]]["answerable"]]

    latencies = [t["latency_ms"] for t in ok_traces]
    summary = {
        "model": config.model,
        "day6_strategy": day6_strategy,
        "embedding_model": embedding_model,
        "config": {
            "default_mode": config.default_mode,
            "max_searches": config.max_searches,
            "top_k": config.top_k,
            "context_budget": config.context_budget,
            "context_mode": config.context_mode,
            "allow_agent_filter": config.allow_agent_filter,
        },
        "total_cases": len(cases),
        "successful": len(ok_traces),
        "errors": len(err_traces),
        "answered": sum(1 for t in ok_traces if t["status"] == "answered"),
        "abstained": sum(1 for t in ok_traces if t["status"] == "abstain"),
        "answerable_context_evidence": f"{ctx_hits}/{len(ans_ok)}",
        "answerable_answered": f"{sum(1 for t in ans_ok if t['status'] == 'answered')}/{len(ans_ok)}",
        "unanswerable_abstained": f"{sum(1 for t in unans_ok if t['status'] == 'abstain')}/{len(unans_ok)}",
        "tool_use": f"{sum(1 for t in ok_traces if t['search_calls'] > 0)}/{len(ok_traces)}",
        "avg_search_calls": round(
            sum(t["search_calls"] for t in ok_traces) / len(ok_traces), 2
        )
        if ok_traces
        else 0.0,
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
    }
    return {"summary": summary, "traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Agentic RAG on Day 7 cases.")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--cases", type=Path, default=Path("starter/data/converted/agent_cases.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional subset of case IDs.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mode", choices=("hybrid", "dense", "bm25"), default="hybrid")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2500)
    parser.add_argument("--context-mode", choices=("score_ranked", "first_come"), default="score_ranked")
    parser.add_argument("--no-agent-filter", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == args.split]
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        raise ValueError("No matching cases selected")

    config = AgentConfig(
        model=args.model or settings.agent_model,
        default_mode=args.mode,
        max_searches=args.max_searches,
        top_k=args.top_k,
        context_budget=args.budget,
        context_mode=args.context_mode,
        allow_agent_filter=not args.no_agent_filter,
    )
    result = run_batch(
        cases,
        chunks,
        config,
        day6_strategy=settings.day6_strategy,
        database_url=settings.database_url,
        openai_api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(result["traces"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "run_summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
