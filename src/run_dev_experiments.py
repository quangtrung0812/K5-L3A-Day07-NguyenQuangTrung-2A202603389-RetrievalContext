"""Run or summarize baseline + improvement agent configurations on the full dev split (17 cases)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agent import AgentConfig
from .config import load_settings
from .run_agent_batch import run_batch

DEV_EXPERIMENTS: list[tuple[str, str, AgentConfig]] = [
    (
        "cfg0_baseline_dense_800",
        "Baseline matching starter defaults: dense pgvector only, max_searches=2, top_k=3, budget=800, first_come, no agent-inferred filter.",
        AgentConfig(
            model="gpt-6-luna",
            default_mode="dense",
            max_searches=2,
            top_k=3,
            context_budget=800,
            context_mode="first_come",
            allow_agent_filter=False,
        ),
    ),
    (
        "cfg1_hybrid_2500_ranked",
        "Improvement 1: Hybrid RRF (dense+BM25), max_searches=3, top_k=3, budget=2500, score_ranked context, agent filter allowed.",
        AgentConfig(
            model="gpt-6-luna",
            default_mode="hybrid",
            max_searches=3,
            top_k=3,
            context_budget=2500,
            context_mode="score_ranked",
            allow_agent_filter=True,
        ),
    ),
    (
        "cfg2_hybrid_4000_k5",
        "Improvement 2: Wider recall & budget: Hybrid RRF, max_searches=4, top_k=5, budget=4000, score_ranked context, agent filter allowed.",
        AgentConfig(
            model="gpt-6-luna",
            default_mode="hybrid",
            max_searches=4,
            top_k=5,
            context_budget=4000,
            context_mode="score_ranked",
            allow_agent_filter=True,
        ),
    ),
    (
        "cfg4_hybrid_4000_k5_followup",
        "Improvement 3 (Selected): Same Hybrid RRF + k=5 + budget=4000 + citation self-correction + follow-up search prompt guidance.",
        AgentConfig(
            model="gpt-6-luna",
            default_mode="hybrid",
            max_searches=4,
            top_k=5,
            context_budget=4000,
            context_mode="score_ranked",
            allow_agent_filter=True,
        ),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize dev experiments.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only matching experiment names.")
    args = parser.parse_args()

    settings = load_settings()
    chunks = json.loads(Path("starter/data/converted/chunks.json").read_text(encoding="utf-8"))
    all_cases = json.loads(Path("starter/data/converted/agent_cases.json").read_text(encoding="utf-8"))
    dev_cases = [c for c in all_cases if c["split"] == "dev"]

    out_root = Path("starter/results/dev-agent-experiments")
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else {name for name, _, _ in DEV_EXPERIMENTS}
    comparison: list[dict[str, Any]] = []
    for name, hypothesis, cfg in DEV_EXPERIMENTS:
        if name not in wanted:
            continue
        cfg.model = settings.agent_model
        res = run_batch(
            dev_cases,
            chunks,
            cfg,
            day6_strategy=settings.day6_strategy,
            database_url=settings.database_url,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.embedding_model,
        )
        exp_dir = out_root / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "traces.json").write_text(
            json.dumps(res["traces"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary = {"experiment": name, "hypothesis": hypothesis, **res["summary"]}
        (exp_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        comparison.append(summary)

    (out_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
