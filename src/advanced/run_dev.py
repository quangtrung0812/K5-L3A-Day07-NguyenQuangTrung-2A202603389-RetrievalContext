"""Run dev benchmark ablation across 3 configurations for WEF Global Risks Report 2024."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..config import load_settings
from ..embeddings import OpenAIQueryEmbedder
from .agent import AdvancedAgentConfig, run_advanced_agent
from .retriever import AdvancedHybridRetriever


DEV_CONFIGS: dict[str, AdvancedAgentConfig] = {
    "cfg0_text_only_baseline": AdvancedAgentConfig(
        name="cfg0_text_only_baseline",
        model="gpt-6-luna",
        default_mode="hybrid",
        text_only=True,
        enable_live_vision=False,
        max_searches=4,
        max_inspects=0,
        top_k=5,
        context_budget=6000,
    ),
    "cfg1_multimodal_no_live_vision": AdvancedAgentConfig(
        name="cfg1_multimodal_no_live_vision",
        model="gpt-6-luna",
        default_mode="hybrid",
        text_only=False,
        enable_live_vision=False,
        max_searches=5,
        max_inspects=0,
        top_k=5,
        context_budget=8000,
    ),
    "cfg2_multimodal_live_vision": AdvancedAgentConfig(
        name="cfg2_multimodal_live_vision",
        model="gpt-6-luna",
        default_mode="hybrid",
        text_only=False,
        enable_live_vision=True,
        max_searches=6,
        max_inspects=4,
        top_k=6,
        context_budget=9000,
    ),
}


def run_config_on_cases(
    cases: list[dict[str, Any]],
    cfg: AdvancedAgentConfig,
    retriever: AdvancedHybridRetriever,
    client: OpenAI,
    embedder: OpenAIQueryEmbedder,
    out_dir: Path,
    index_dir: Path = Path("data/advanced/index"),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    traces: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        trace = run_advanced_agent(
            question=case["question"],
            retriever=retriever,
            client=client,  # type: ignore[arg-type]
            config=cfg,
            query_id=case["id"],
            index_dir=index_dir,
            embedder=embedder,
        )
        trace["category"] = case["category"]
        trace["answerable"] = case["answerable"]
        trace["evidence_pages"] = case["evidence_pages"]
        trace["gold_answer"] = case["gold_answer"]
        traces.append(trace)
        print(
            f"[{cfg.name}] {idx:02d}/{len(cases)} {case['id']} ({case['category']}) -> "
            f"status={trace['status']} inspect={trace['inspected_pages']} "
            f"lat={trace['latency_ms']}ms cost=${trace['cost_usd']:.6f}",
            flush=True,
        )

    (out_dir / "traces.json").write_text(
        json.dumps({"config": cfg.__dict__, "traces": traces}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_cost = round(sum(t["cost_usd"] for t in traces), 6)
    avg_lat = round(sum(t["latency_ms"] for t in traces) / max(len(traces), 1), 2)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"answered": 0, "abstain": 0, "total": 0})
    for t in traces:
        by_cat[t["category"]]["total"] += 1
        by_cat[t["category"]][t["status"]] = by_cat[t["category"]].get(t["status"], 0) + 1

    summary = {
        "config_name": cfg.name,
        "cases": len(traces),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstain": sum(1 for t in traces if t["status"] == "abstain"),
        "total_cost_usd": total_cost,
        "avg_latency_ms": avg_lat,
        "by_category_status": dict(by_cat),
    }
    (out_dir / "summary_raw.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dev ablation on WEF Global Risks Report 2024.")
    parser.add_argument("--configs", nargs="*", default=list(DEV_CONFIGS.keys()))
    args = parser.parse_args()

    settings = load_settings()
    embedder = OpenAIQueryEmbedder(api_key=settings.openai_api_key, model_name=settings.embedding_model)
    retriever = AdvancedHybridRetriever.from_index_dir(
        index_dir=Path("data/advanced/index"),
        database_url=settings.database_url,
        embedder=embedder,
    )
    client = OpenAI(api_key=settings.openai_api_key)
    dev_data = json.loads(Path("data/advanced/benchmark/dev.json").read_text(encoding="utf-8"))
    cases = dev_data["cases"]

    summaries = []
    for name in args.configs:
        cfg = DEV_CONFIGS[name]
        if settings.agent_model and cfg.model != settings.agent_model:
            cfg = AdvancedAgentConfig(**{**cfg.__dict__, "model": settings.agent_model})
        s = run_config_on_cases(
            cases=cases,
            cfg=cfg,
            retriever=retriever,
            client=client,
            embedder=embedder,
            out_dir=Path("data/advanced/results/dev") / cfg.name,
        )
        summaries.append(s)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
