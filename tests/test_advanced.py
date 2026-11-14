"""Unit tests for the advanced 100+ page multimodal RAG pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.advanced.agent import AdvancedAgentConfig, run_advanced_agent
from src.advanced.retriever import AdvancedHybridRetriever


def _make_fake_response(content: str = "", tool_calls: list[dict] | None = None, usage=(100, 30)):
    tcs = []
    for tc in tool_calls or []:
        tcs.append(
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(name=tc["name"], arguments=json.dumps(tc["args"])),
            )
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tcs))],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
    )


class FakeAdvancedClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class AdvancedPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "id": "wef-grr-2024:p009:visual:01",
                "source_id": "wef-grr-2024",
                "source": "WEF Global Risks Report 2024",
                "page": 9,
                "modality": "visual",
                "figure_id": "FIGURE D",
                "bbox": [20.0, 60.0, 560.0, 480.0],
                "crop_path": "data/advanced/index/crops/p009_visual.png",
                "page_image_path": "data/advanced/index/pages/p009.png",
                "text": "[Visual Index p9 | FIGURE D] Global risks interconnections map.",
                "metadata": {"page": 9, "modality": "visual"},
            },
            {
                "id": "wef-grr-2024:p010:text:01",
                "source_id": "wef-grr-2024",
                "source": "WEF Global Risks Report 2024",
                "page": 10,
                "modality": "text",
                "figure_id": None,
                "bbox": [40.0, 80.0, 520.0, 300.0],
                "crop_path": None,
                "page_image_path": "data/advanced/index/pages/p010.png",
                "text": "Extreme weather events lead environmental concerns over 10 years.",
                "metadata": {"page": 10, "modality": "text"},
            },
        ]

    def test_bm25_modality_and_page_scope_filtering(self) -> None:
        ret = AdvancedHybridRetriever(chunks=self.chunks)
        vis = ret.search("Global risks interconnections map", mode="bm25", top_k=3, modality="visual")
        self.assertEqual([r["id"] for r in vis], ["wef-grr-2024:p009:visual:01"])
        txt = ret.search("Extreme weather", mode="bm25", top_k=3, page_start=10, page_end=10)
        self.assertEqual([r["id"] for r in txt], ["wef-grr-2024:p010:text:01"])

    def test_live_vision_inspect_attaches_image_and_allows_citation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idx_dir = Path(tmp)
            (idx_dir / "crops").mkdir(parents=True)
            (idx_dir / "crops" / "p009_visual.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

            ret = AdvancedHybridRetriever(chunks=self.chunks)
            client = FakeAdvancedClient(
                [
                    _make_fake_response(
                        tool_calls=[
                            {"id": "t1", "name": "search_documents", "args": {"query": "interconnections map", "mode": "bm25"}}
                        ]
                    ),
                    _make_fake_response(
                        tool_calls=[
                            {"id": "t2", "name": "inspect_page_image", "args": {"page": 9, "focus": "crop"}}
                        ]
                    ),
                    _make_fake_response(
                        tool_calls=[
                            {
                                "id": "t3",
                                "name": "submit_answer",
                                "args": {
                                    "answer": "Societal polarization and Economic downturn are largest nodes [wef-grr-2024:p009:image].",
                                    "citations": ["wef-grr-2024:p009:image"],
                                },
                            }
                        ]
                    ),
                ]
            )
            cfg = AdvancedAgentConfig(name="test_vis", enable_live_vision=True)
            res = run_advanced_agent(
                "What are the largest nodes in Figure D?",
                retriever=ret,
                client=client,  # type: ignore[arg-type]
                config=cfg,
                query_id="q_vis",
                index_dir=idx_dir,
            )
            self.assertEqual(res["status"], "answered")
            self.assertEqual(res["inspected_pages"], [9])
            self.assertEqual(res["citations"], ["wef-grr-2024:p009:image"])
            # Verify third LLM call received a multimodal user image message
            third_msgs = client.calls[2]["messages"]
            self.assertTrue(any(isinstance(m.get("content"), list) for m in third_msgs))

    def test_abstain_after_search(self) -> None:
        ret = AdvancedHybridRetriever(chunks=self.chunks)
        client = FakeAdvancedClient(
            [
                _make_fake_response(
                    tool_calls=[{"id": "t1", "name": "search_documents", "args": {"query": "missing topic", "mode": "bm25"}}]
                ),
                _make_fake_response(
                    tool_calls=[{"id": "t2", "name": "abstain", "args": {"reason": "Not present in report."}}]
                ),
            ]
        )
        cfg = AdvancedAgentConfig(name="test_abs", enable_live_vision=False)
        res = run_advanced_agent("Missing?", retriever=ret, client=client, config=cfg, query_id="u1")  # type: ignore[arg-type]
        self.assertEqual(res["status"], "abstain")
        self.assertEqual(res["citations"], [])

    def test_benchmark_quotas_and_zero_overlap(self) -> None:
        from collections import Counter

        dev_path = Path("data/advanced/benchmark/dev.json")
        test_path = Path("data/advanced/benchmark/test.json")
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))

        dev_cases = dev["cases"]
        test_cases = test["cases"]
        self.assertGreaterEqual(len(dev_cases), 18)
        self.assertGreaterEqual(len(test_cases), 30)

        dev_counts = Counter(c["category"] for c in dev_cases)
        test_counts = Counter(c["category"] for c in test_cases)
        self.assertGreaterEqual(sum(1 for c in dev_cases if c["answerable"]), 15)
        self.assertGreaterEqual(dev_counts["Thiếu bằng chứng"], 3)

        self.assertGreaterEqual(test_counts["Bảng"], 6)
        self.assertGreaterEqual(test_counts["Ảnh, sơ đồ"], 5)
        self.assertGreaterEqual(test_counts["Biểu đồ"], 5)
        self.assertGreaterEqual(test_counts["Multi-hop"], 4)
        self.assertGreaterEqual(test_counts["Text"], 4)
        self.assertGreaterEqual(test_counts["Thiếu bằng chứng"], 6)

        dev_q = {c["question"].strip().lower() for c in dev_cases}
        test_q = {c["question"].strip().lower() for c in test_cases}
        self.assertEqual(len(dev_q & test_q), 0)
        dev_id = {c["id"] for c in dev_cases}
        test_id = {c["id"] for c in test_cases}
        self.assertEqual(len(dev_id & test_id), 0)
        for c in dev_cases + test_cases:
            for p in c["evidence_pages"]:
                self.assertTrue(1 <= p <= 124, msg=f"Invalid page {p} in {c['id']}")


if __name__ == "__main__":
    unittest.main()
