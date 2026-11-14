"""Prepare Day 7 chunks, retrieval questions, and 24+ agent benchmark cases from Day 6 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DAY6_CHUNKS = Path(
    r"D:\Vin\K5-L3A-Day06-NguyenQuangTrung-2A202603389-DataIngestion\results\chunks_strategy_B.json"
)
DEFAULT_DAY6_QUESTIONS = Path(
    r"D:\Vin\K5-L3A-Day06-NguyenQuangTrung-2A202603389-DataIngestion\benchmark\questions.json"
)
DEFAULT_OUTPUT_DIR = Path("starter/data/converted")

UNANSWERABLE_CASES: list[dict[str, Any]] = [
    {
        "id": "u_dev_01",
        "split": "dev",
        "question": (
            "What are the exact maximum administrative fines and compliance deadlines "
            "specified in the EU AI Act for prohibited AI practices?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The corpus only contains NIST AI RMF 1.0 (2023) and NIST CSF 2.0 (2024); "
            "it does not include the EU AI Act, administrative fines, or enforcement deadlines."
        ),
    },
    {
        "id": "u_dev_02",
        "split": "dev",
        "question": (
            "How does NIST AI RMF 1.0 map its GOVERN subcategories directly to "
            "ISO/IEC 42001:2023 clause numbers?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "NIST AI RMF 1.0 (published January 2023) does not contain any mapping to "
            "ISO/IEC 42001:2023 clause numbers in the indexed corpus."
        ),
    },
    {
        "id": "u_test_01",
        "split": "test",
        "question": (
            "What is the official accreditation and certification fee schedule charged by NIST "
            "for organizations seeking NIST CSF 2.0 compliance certification?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "NIST CSF 2.0 is voluntary guidance; the corpus contains no NIST certification "
            "program or fee schedule."
        ),
    },
    {
        "id": "u_test_02",
        "split": "test",
        "question": (
            "What specific GDPR Article 22 automated decision-making penalties are prescribed "
            "in NIST AI RMF 1.0?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The corpus does not mention GDPR Article 22 or prescribe any regulatory penalties."
        ),
    },
]


def convert_chunk(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw.get("id", raw.get("chunk_id")),
        "source_id": raw.get("source_id", raw.get("document_id")),
        "source": raw.get("source"),
        "page": raw.get("page"),
        "text": raw.get("text"),
        "metadata": raw.get("metadata", {}),
    }


def convert_answerable_question(raw: dict[str, Any]) -> dict[str, Any]:
    gold: list[dict[str, Any]] = []
    if raw.get("gold"):
        gold = list(raw["gold"])
    else:
        gold.append(
            {
                "source_id": raw.get("gold_source_id"),
                "page": raw.get("gold_page"),
                "evidence": raw.get("evidence_quote"),
            }
        )
        for alt in raw.get("alternative_gold") or []:
            gold.append(
                {
                    "source_id": alt.get("gold_source_id", alt.get("source_id")),
                    "page": alt.get("gold_page", alt.get("page")),
                    "evidence": alt.get("evidence_quote", alt.get("evidence")),
                }
            )

    item: dict[str, Any] = {
        "id": raw["id"],
        "split": raw["split"],
        "question": raw["question"],
        "gold": gold,
        "expected_answer": raw["expected_answer"],
    }
    if "filter" in raw and raw["filter"] is not None:
        item["filter"] = raw["filter"]
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Day 7 data & benchmark from Day 6.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_DAY6_CHUNKS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_DAY6_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    raw_chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if not isinstance(raw_chunks, list):
        raise ValueError("Day 6 chunks file must be a JSON array")

    raw_q_payload = json.loads(args.questions.read_text(encoding="utf-8"))
    raw_questions = raw_q_payload["questions"] if isinstance(raw_q_payload, dict) else raw_q_payload
    if not isinstance(raw_questions, list):
        raise ValueError("Day 6 questions payload must contain a list of questions")

    chunks = [convert_chunk(c) for c in raw_chunks]
    questions = [convert_answerable_question(q) for q in raw_questions]

    agent_cases: list[dict[str, Any]] = []
    for q in questions:
        case = {
            **q,
            "answerable": True,
            "expected_behavior": "search_and_answer",
        }
        agent_cases.append(case)
    agent_cases.extend(UNANSWERABLE_CASES)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "agent_cases.json").write_text(
        json.dumps(agent_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dev_ans = sum(1 for c in agent_cases if c["split"] == "dev" and c["answerable"])
    dev_unans = sum(1 for c in agent_cases if c["split"] == "dev" and not c["answerable"])
    test_ans = sum(1 for c in agent_cases if c["split"] == "test" and c["answerable"])
    test_unans = sum(1 for c in agent_cases if c["split"] == "test" and not c["answerable"])

    print(
        f"Prepared {len(chunks)} chunks, {len(questions)} retrieval questions, "
        f"{len(agent_cases)} agent cases "
        f"(dev: {dev_ans} answerable + {dev_unans} unanswerable; "
        f"test: {test_ans} answerable + {test_unans} unanswerable)."
    )


if __name__ == "__main__":
    main()
