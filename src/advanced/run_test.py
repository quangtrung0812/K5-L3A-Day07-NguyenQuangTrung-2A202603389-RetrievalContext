"""Run the locked multimodal pipeline once on the held-out 30-case test benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from ..config import load_settings
from ..embeddings import OpenAIQueryEmbedder
from .agent import AdvancedAgentConfig
from .retriever import AdvancedHybridRetriever
from .run_dev import DEV_CONFIGS, run_config_on_cases


def main() -> None:
    out_dir = Path("data/advanced/results/test/cfg2_multimodal_live_vision")
    traces_file = out_dir / "traces.json"
    if traces_file.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing test run at {traces_file}. "
            "Test benchmark may only be executed once after locking pipeline on dev."
        )

    settings = load_settings()
    base_cfg = DEV_CONFIGS["cfg2_multimodal_live_vision"]
    cfg = AdvancedAgentConfig(**{**base_cfg.__dict__, "model": settings.agent_model or base_cfg.model})

    embedder = OpenAIQueryEmbedder(api_key=settings.openai_api_key, model_name=settings.embedding_model)
    retriever = AdvancedHybridRetriever.from_index_dir(
        index_dir=Path("data/advanced/index"),
        database_url=settings.database_url,
        embedder=embedder,
    )
    client = OpenAI(api_key=settings.openai_api_key)
    test_data = json.loads(Path("data/advanced/benchmark/test.json").read_text(encoding="utf-8"))
    summary = run_config_on_cases(
        cases=test_data["cases"],
        cfg=cfg,
        retriever=retriever,
        client=client,
        embedder=embedder,
        out_dir=out_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
