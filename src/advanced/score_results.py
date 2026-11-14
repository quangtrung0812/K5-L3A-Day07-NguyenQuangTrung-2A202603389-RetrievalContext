"""Score reviewed dev/test traces with strict denominator accounting (never drops errors)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def compute_summary(traces_path: Path, reviews_path: Path) -> dict[str, Any]:
    traces_payload = json.loads(traces_path.read_text(encoding="utf-8"))
    traces: list[dict[str, Any]] = traces_payload["traces"]
    reviews_payload = json.loads(reviews_path.read_text(encoding="utf-8"))
    reviews_by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in reviews_payload["reviews"]}

    if len(reviews_by_id) != len(traces):
        missing = [t["query_id"] for t in traces if t["query_id"] not in reviews_by_id]
        raise ValueError(f"Reviews count ({len(reviews_by_id)}) != traces ({len(traces)}). Missing: {missing}")

    passed = 0
    failed_cases: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
    latencies: list[float] = []
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_embedding_tokens = 0
    total_inspect_calls = 0

    for t in traces:
        qid = t["query_id"]
        rev = reviews_by_id[qid]
        verdict = rev["verdict"]
        if verdict not in ("pass", "fail"):
            raise ValueError(f"Invalid verdict {verdict!r} for {qid}; expected 'pass' or 'fail'.")
        cat = t["category"]
        by_category[cat]["total"] += 1
        if verdict == "pass":
            passed += 1
            by_category[cat]["passed"] += 1
        else:
            by_category[cat]["failed"] += 1
            failed_cases.append(
                {
                    "id": qid,
                    "category": cat,
                    "question": t["question"],
                    "status": t["status"],
                    "citations": t["citations"],
                    "inspected_pages": t["inspected_pages"],
                    "gold_answer": t.get("gold_answer"),
                    "agent_answer": t["answer"],
                    "failure_reason": rev.get("notes", ""),
                }
            )
        latencies.append(float(t["latency_ms"]))
        total_cost += float(t["cost_usd"])
        total_prompt_tokens += int(t["llm_prompt_tokens"])
        total_completion_tokens += int(t["llm_completion_tokens"])
        total_embedding_tokens += int(t["embedding_tokens"])
        total_inspect_calls += int(t.get("inspect_calls", 0))

    total = len(traces)
    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat))) if sorted_lat else 0
    cat_summary = {
        cat: {
            "passed": stats["passed"],
            "total": stats["total"],
            "score_str": f"{stats['passed']}/{stats['total']}",
            "accuracy_pct": round(100.0 * stats["passed"] / stats["total"], 2),
        }
        for cat, stats in by_category.items()
    }
    return {
        "config": traces_payload.get("config", {}),
        "overall": {
            "passed": passed,
            "total": total,
            "score_str": f"{passed}/{total}",
            "accuracy_pct": round(100.0 * passed / total, 2) if total else 0.0,
        },
        "by_category": cat_summary,
        "telemetry": {
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_query_usd": round(total_cost / total, 6) if total else 0.0,
            "avg_latency_ms": round(sum(latencies) / total, 2) if total else 0.0,
            "p95_latency_ms": round(sorted_lat[p95_idx], 2) if sorted_lat else 0.0,
            "total_llm_prompt_tokens": total_prompt_tokens,
            "total_llm_completion_tokens": total_completion_tokens,
            "total_embedding_tokens": total_embedding_tokens,
            "total_inspect_calls": total_inspect_calls,
        },
        "failed_cases": failed_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score reviewed RAG traces.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = compute_summary(args.traces, args.reviews)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False))
    print(json.dumps(summary["by_category"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
