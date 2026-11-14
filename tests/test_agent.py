"""Unit tests for Agentic RAG controller, context selection, citation check, and abstention."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from src.agent import AgentConfig, run_agentic_rag, select_context

CHUNKS = [
    {
        "id": "doc:p1:c1",
        "source_id": "doc",
        "source": "doc.pdf",
        "page": 1,
        "text": "Trustworthy AI includes validity, safety, security, accountability, and fairness.",
        "metadata": {"domain": "ai_risk", "year": 2023},
        "score": 0.45,
        "rank": 2,
    },
    {
        "id": "doc:p2:c1",
        "source_id": "doc",
        "source": "doc.pdf",
        "page": 2,
        "text": "GOVERN 6 covers third-party governance and contingency plans for supplier failures.",
        "metadata": {"domain": "ai_risk", "year": 2023},
        "score": 0.92,
        "rank": 1,
    },
]


class _FakeRetriever:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def search(self, query, *, mode="hybrid", top_k=3, filters=None):
        self.calls.append((query, mode, top_k, filters))
        if self.batches:
            return self.batches.pop(0)
        return []


def _make_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _make_response(tool_calls=None, content="", prompt_tokens=100, completion_tokens=25):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _FakeOpenAIClient:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        return next(self._responses)


class AgenticRagTests(unittest.TestCase):
    def test_multi_turn_search_and_valid_citation(self):
        client = _FakeOpenAIClient(
            [
                _make_response(
                    [_make_tool_call("c1", "search_documents", {"query": "GOVERN 6", "mode": "hybrid"})]
                ),
                _make_response(
                    [
                        _make_tool_call(
                            "c2",
                            "submit_answer",
                            {
                                "answer": "GOVERN 6 covers third-party governance [doc:p2:c1].",
                                "citations": ["doc:p2:c1"],
                            },
                        )
                    ]
                ),
            ]
        )
        retriever = _FakeRetriever([[CHUNKS[1]]])
        res = run_agentic_rag(
            "What does GOVERN 6 cover?",
            client=client,
            retriever=retriever,
            config=AgentConfig(max_searches=3, context_budget=1000),
            user_filters={"domain": "ai_risk"},
            query_id="t1",
        )
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["doc:p2:c1"])
        self.assertEqual(res["search_calls"], 1)
        self.assertEqual(retriever.calls[0][3], {"domain": "ai_risk"})

    def test_abstain_tool_after_search(self):
        client = _FakeOpenAIClient(
            [
                _make_response(
                    [_make_tool_call("c1", "search_documents", {"query": "EU AI Act fines", "mode": "dense"})]
                ),
                _make_response(
                    [_make_tool_call("c2", "abstain", {"reason": "No EU AI Act content in corpus."})]
                ),
            ]
        )
        retriever = _FakeRetriever([[CHUNKS[0]]])
        res = run_agentic_rag(
            "What are EU AI Act fines?",
            client=client,
            retriever=retriever,
            config=AgentConfig(),
            query_id="t2",
        )
        self.assertEqual(res["status"], "abstain")
        self.assertEqual(res["citations"], [])
        self.assertEqual(res["search_calls"], 1)

    def test_invalid_citation_falls_back_to_abstain(self):
        client = _FakeOpenAIClient(
            [
                _make_response([_make_tool_call("c1", "search_documents", {"query": "AI"})]),
                _make_response(
                    [
                        _make_tool_call(
                            "c2",
                            "submit_answer",
                            {"answer": "Claim [missing:id].", "citations": ["missing:id"]},
                        )
                    ]
                ),
                _make_response(
                    [
                        _make_tool_call(
                            "c3",
                            "submit_answer",
                            {"answer": "Still wrong [missing:id].", "citations": ["missing:id"]},
                        )
                    ]
                ),
            ]
        )
        retriever = _FakeRetriever([[CHUNKS[0]]])
        res = run_agentic_rag(
            "What is AI?",
            client=client,
            retriever=retriever,
            config=AgentConfig(),
        )
        self.assertEqual(res["status"], "abstain")
        self.assertEqual(res["events"][-1]["event"], "invalid_or_missing_citations")

    def test_citation_self_correction_succeeds(self):
        client = _FakeOpenAIClient(
            [
                _make_response([_make_tool_call("c1", "search_documents", {"query": "AI"})]),
                _make_response(
                    [
                        _make_tool_call(
                            "c2",
                            "submit_answer",
                            {"answer": "Claim without inline tag.", "citations": ["doc:p1:c1"]},
                        )
                    ]
                ),
                _make_response(
                    [
                        _make_tool_call(
                            "c3",
                            "submit_answer",
                            {"answer": "Claim with tag [doc:p1:c1].", "citations": ["doc:p1:c1"]},
                        )
                    ]
                ),
            ]
        )
        retriever = _FakeRetriever([[CHUNKS[0]]])
        res = run_agentic_rag(
            "What is AI?",
            client=client,
            retriever=retriever,
            config=AgentConfig(),
        )
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["doc:p1:c1"])

    def test_select_context_score_ranked_prefers_higher_score(self):
        ctx, ids, _ = select_context(CHUNKS, budget=160, mode="score_ranked")
        self.assertEqual(ids, ["doc:p2:c1"])
        self.assertLessEqual(len(ctx), 160)


if __name__ == "__main__":
    unittest.main()
