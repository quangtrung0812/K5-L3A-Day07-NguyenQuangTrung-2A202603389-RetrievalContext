"""Multimodal Agentic RAG for 124-page WEF Global Risks Report 2024."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from ..agent import AgentCostTracker, OpenAIChatClient
from ..embeddings import OpenAIQueryEmbedder
from .retriever import AdvancedHybridRetriever

ADVANCED_SYSTEM_PROMPT = """You are an exact, evidence-grounded research assistant analyzing the WEF Global Risks Report 2024 (124 pages).

Core rules:
1. Always call `search_documents` first to locate relevant pages across text, tables, and visual figures.
2. When a question involves ANY chart, diagram, network map, world map, box, or multi-column table (or when extracted text/summaries could omit exact values, axis units, rankings, colors, or edges), call `inspect_page_image` with `focus="full"` on the candidate page(s) so axis labels, legends, margin annotations, and footnotes are completely unclipped. Read every label word-for-word (e.g. distinguish `frontier technologies` vs `AI technologies`) and check callout notes.
3. For multi-hop questions spanning multiple pages/figures/tables, inspect or search ALL needed pages before calling `submit_answer`.
4. Read numbers, percentages, rankings, units, ranges, years, and country names with complete precision from the retrieved chunks and inspected page images. Do not guess or round beyond what the source states.
5. Citations rule: cite every factual claim inline using exact IDs from your context/inspected pages, e.g. [wef-grr-2024:p009:visual:01] or [wef-grr-2024:p009:image], and pass the exact list of cited IDs to `submit_answer`.
6. Strict Scope & Abstention rule: before answering, verify that the EXACT edition/year, entity, metric, framework, or statistic asked for explicitly appears in the WEF Global Risks Report 2024. NEVER substitute another edition/year (e.g. do NOT use 2024 survey rankings to answer a question asking for Global Risks Report 2025) or a loosely related topic. If the exact requested item is missing, run at least 2 distinct searches (e.g. hybrid + bm25) to confirm its absence, and then call `abstain` — NEVER call `submit_answer` to say evidence is missing or to guess from another year/topic.
"""


@dataclass(frozen=True)
class AdvancedAgentConfig:
    name: str
    model: str = "gpt-6-luna"
    default_mode: str = "hybrid"
    text_only: bool = False
    enable_live_vision: bool = True
    max_searches: int = 5
    max_inspects: int = 4
    top_k: int = 5
    context_budget: int = 5000
    temperature: float = 0.0


def _build_tools(enable_live_vision: bool, text_only: bool) -> list[dict[str, Any]]:
    search_props: dict[str, Any] = {
        "query": {"type": "string", "description": "Focused search query."},
        "mode": {
            "type": "string",
            "enum": ["hybrid", "dense", "bm25"],
            "description": "Retrieval mode.",
        },
        "page_start": {"type": "integer", "description": "Optional min page (1..124)."},
        "page_end": {"type": "integer", "description": "Optional max page (1..124)."},
    }
    if not text_only:
        search_props["modality"] = {
            "type": "string",
            "enum": ["text", "table", "visual"],
            "description": "Optional modality filter.",
        }

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the 124-page WEF Global Risks Report 2024 index.",
                "parameters": {
                    "type": "object",
                    "properties": search_props,
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit the final verified answer with exact inline [chunk_id] citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Complete, exact grounded answer with [id] markers."},
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chunk IDs or inspected page image IDs supporting the answer.",
                        },
                    },
                    "required": ["answer", "citations"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "abstain",
                "description": "Abstain when the report lacks sufficient evidence after multiple searches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why evidence is missing."}
                    },
                    "required": ["reason"],
                },
            },
        },
    ]
    if enable_live_vision:
        tools.insert(
            1,
            {
                "type": "function",
                "function": {
                    "name": "inspect_page_image",
                    "description": (
                        "Open the original high-resolution rendered PNG of a PDF page (or cropped figure/table) "
                        "to read exact chart numbers, axes, units, colors, network links, maps, or tables."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer", "description": "PDF page number (1..124)."},
                            "focus": {
                                "type": "string",
                                "enum": ["full", "crop"],
                                "description": "'full' for the complete unclipped page (recommended), 'crop' for cropped figure.",
                            },
                        },
                        "required": ["page"],
                    },
                },
            },
        )
    return tools


def _pack_results(candidates: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in candidates:
        cid = item["id"]
        if cid not in seen:
            seen.add(cid)
            unique.append(item)

    blocks: list[str] = []
    ids: list[str] = []
    for item in unique:
        fig = f" | {item['figure_id']}" if item.get("figure_id") else ""
        bbox = f" | bbox={item['bbox']}" if item.get("bbox") else ""
        header = f"[{item['id']} | page {item['page']} | {item.get('modality', 'text')}{fig}{bbox}]"
        block = f"{header}\n{item['text']}"
        candidate_text = "\n\n".join(blocks + [block])
        if len(candidate_text) <= budget or not blocks:
            blocks.append(block)
            ids.append(item["id"])
    return "\n\n".join(blocks), ids


def _resolve_image_path(page: int, focus: str, index_dir: Path) -> Path:
    crop_path = index_dir / "crops" / f"p{page:03d}_visual.png"
    full_path = index_dir / "pages" / f"p{page:03d}.png"
    if focus == "crop" and crop_path.exists():
        return crop_path
    if full_path.exists():
        return full_path
    if crop_path.exists():
        return crop_path
    raise FileNotFoundError(f"No rendered PNG found for page {page} in {index_dir}")


def run_advanced_agent(
    question: str,
    retriever: AdvancedHybridRetriever,
    client: OpenAIChatClient,
    config: AdvancedAgentConfig,
    query_id: str = "",
    index_dir: Path = Path("data/advanced/index"),
    embedder: OpenAIQueryEmbedder | None = None,
) -> dict[str, Any]:
    """Run multimodal agent loop supporting hybrid retrieval + live page image inspection."""
    t_start = time.perf_counter()
    tracker = AgentCostTracker(model=config.model)
    emb_tokens_start = embedder.total_prompt_tokens if embedder else 0
    emb_cost_start = embedder.total_cost_usd if embedder else 0.0

    tools = _build_tools(config.enable_live_vision, config.text_only)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ADVANCED_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question.strip()}"},
    ]

    candidates: list[dict[str, Any]] = []
    context_ids: list[str] = []
    shown_ids: list[str] = []
    inspected_pages: list[int] = []
    inspected_ids: list[str] = []
    events: list[dict[str, Any]] = []
    search_calls = 0
    inspect_calls = 0
    citation_retries = 0

    max_turns = config.max_searches + config.max_inspects + 4
    for _turn in range(max_turns):
        t_llm = time.perf_counter()
        create_kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": config.temperature,
        }
        if config.model.startswith(("gpt-5", "gpt-6")):
            create_kwargs["reasoning_effort"] = "none"
        response = client.chat.completions.create(**create_kwargs)
        llm_ms = round((time.perf_counter() - t_llm) * 1000.0, 2)
        tracker.llm_calls += 1
        tracker.add_usage(getattr(response, "usage", None))

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            text = (msg.content or "").strip()
            events.append({"event": "plain_response", "llm_ms": llm_ms, "content": text})
            if search_calls == 0:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "Call `search_documents` first."})
                continue
            return _finish_trace(
                query_id=query_id,
                question=question,
                status="abstain",
                answer="Insufficient evidence in retrieved context to answer reliably.",
                citations=[],
                search_calls=search_calls,
                inspect_calls=inspect_calls,
                inspected_pages=inspected_pages,
                context_ids=context_ids,
                events=events + [{"event": "abstain", "reason": "no_tool_call"}],
                tracker=tracker,
                embedder=embedder,
                emb_tokens_start=emb_tokens_start,
                emb_cost_start=emb_cost_start,
                elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        pending_images: list[tuple[int, str, Path]] = []

        for tc in tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            if fn_name == "search_documents":
                if search_calls >= config.max_searches:
                    events.append({"event": "search_limit", "llm_ms": llm_ms})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(
                                {
                                    "error": "Search limit reached; inspect retrieved pages or call submit_answer/abstain.",
                                    "valid_ids": sorted(set(shown_ids + inspected_ids)),
                                }
                            ),
                        }
                    )
                    continue

                query = (args.get("query") or "").strip()
                mode = args.get("mode") or config.default_mode
                if mode not in ("hybrid", "dense", "bm25"):
                    mode = config.default_mode
                modality = "text" if config.text_only else args.get("modality")
                page_start = args.get("page_start")
                page_end = args.get("page_end")

                t_ret = time.perf_counter()
                res = retriever.search(
                    query,
                    mode=mode,
                    top_k=config.top_k,
                    modality=modality,
                    page_start=page_start,
                    page_end=page_end,
                )
                ret_ms = round((time.perf_counter() - t_ret) * 1000.0, 2)
                search_calls += 1
                candidates = list(res) + candidates
                context_text, context_ids = _pack_results(candidates, budget=config.context_budget)
                for cid in context_ids:
                    if cid not in shown_ids:
                        shown_ids.append(cid)

                events.append(
                    {
                        "event": "search",
                        "mode": mode,
                        "query": query,
                        "modality": modality,
                        "page_range": [page_start, page_end],
                        "llm_ms": llm_ms,
                        "retrieval_ms": ret_ms,
                        "returned_ids": [r["id"] for r in res],
                        "pages": [r["page"] for r in res],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "search_calls_used": search_calls,
                                "searches_remaining": config.max_searches - search_calls,
                                "valid_ids": sorted(set(shown_ids + inspected_ids)),
                                "packed_context": context_text,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

            elif fn_name == "inspect_page_image" and config.enable_live_vision:
                if inspect_calls >= config.max_inspects:
                    events.append({"event": "inspect_limit", "llm_ms": llm_ms})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": "Inspect limit reached; answer or abstain using available evidence."}),
                        }
                    )
                    continue
                page = max(1, min(124, int(args.get("page", 1))))
                focus = args.get("focus") or "full"
                if focus not in ("crop", "full"):
                    focus = "full"
                img_path = _resolve_image_path(page, focus, index_dir)
                inspect_calls += 1
                if page not in inspected_pages:
                    inspected_pages.append(page)
                img_id = f"wef-grr-2024:p{page:03d}:image"
                if img_id not in inspected_ids:
                    inspected_ids.append(img_id)
                events.append(
                    {
                        "event": "inspect_page_image",
                        "page": page,
                        "focus": focus,
                        "image_path": str(img_path).replace("\\", "/"),
                        "citation_id": img_id,
                        "llm_ms": llm_ms,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "ok": True,
                                "page": page,
                                "focus": focus,
                                "citation_id": img_id,
                                "note": "The high-resolution page image is attached in the next user message.",
                            }
                        ),
                    }
                )
                pending_images.append((page, img_id, img_path))

            elif fn_name == "abstain":
                if search_calls == 0:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": "Call search_documents before abstaining."}),
                        }
                    )
                    continue
                reason = (args.get("reason") or "").strip() or "Insufficient evidence."
                events.append({"event": "abstain", "llm_ms": llm_ms, "reason": reason})
                return _finish_trace(
                    query_id=query_id,
                    question=question,
                    status="abstain",
                    answer=f"Insufficient evidence in the provided report: {reason}",
                    citations=[],
                    search_calls=search_calls,
                    inspect_calls=inspect_calls,
                    inspected_pages=inspected_pages,
                    context_ids=context_ids,
                    events=events,
                    tracker=tracker,
                    embedder=embedder,
                    emb_tokens_start=emb_tokens_start,
                    emb_cost_start=emb_cost_start,
                    elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
                )

            elif fn_name == "submit_answer":
                answer = (args.get("answer") or "").strip()
                raw_cits = args.get("citations") or []
                valid_set = set(shown_ids + inspected_ids)
                valid_cits = [c for c in raw_cits if c in valid_set]
                invalid_cits = [c for c in raw_cits if c not in valid_set]
                if not answer or not valid_cits or invalid_cits:
                    if citation_retries < 1:
                        citation_retries += 1
                        events.append(
                            {
                                "event": "citation_retry",
                                "invalid_citations": invalid_cits,
                                "llm_ms": llm_ms,
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(
                                    {
                                        "error": (
                                            f"Invalid or empty citations {invalid_cits}. "
                                            f"You MUST only cite IDs from valid_ids={sorted(valid_set)}."
                                        )
                                    }
                                ),
                            }
                        )
                        continue
                    events.append({"event": "abstain", "reason": "invalid_citations_after_retry", "llm_ms": llm_ms})
                    return _finish_trace(
                        query_id=query_id,
                        question=question,
                        status="abstain",
                        answer="Insufficient grounded citations in context.",
                        citations=[],
                        search_calls=search_calls,
                        inspect_calls=inspect_calls,
                        inspected_pages=inspected_pages,
                        context_ids=context_ids,
                        events=events,
                        tracker=tracker,
                        embedder=embedder,
                        emb_tokens_start=emb_tokens_start,
                        emb_cost_start=emb_cost_start,
                        elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
                    )
                events.append({"event": "answer", "citations": valid_cits, "llm_ms": llm_ms})
                return _finish_trace(
                    query_id=query_id,
                    question=question,
                    status="answered",
                    answer=answer,
                    citations=list(dict.fromkeys(valid_cits)),
                    search_calls=search_calls,
                    inspect_calls=inspect_calls,
                    inspected_pages=inspected_pages,
                    context_ids=context_ids,
                    events=events,
                    tracker=tracker,
                    embedder=embedder,
                    emb_tokens_start=emb_tokens_start,
                    emb_cost_start=emb_cost_start,
                    elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            else:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown/disabled tool: {fn_name}"}),
                    }
                )

        for page, img_id, img_path in pending_images:
            b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Attached original rendered PNG for page {page} (citation ID: [{img_id}]). "
                                "Read exact values, labels, axes, colors, links, or table rows directly from this image."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + b64, "detail": "high"},
                        },
                    ],
                }
            )

    events.append({"event": "abstain", "reason": "max_turns_reached"})
    return _finish_trace(
        query_id=query_id,
        question=question,
        status="abstain",
        answer="Stopped after reaching max agent turns without a final verified answer.",
        citations=[],
        search_calls=search_calls,
        inspect_calls=inspect_calls,
        inspected_pages=inspected_pages,
        context_ids=context_ids,
        events=events,
        tracker=tracker,
        embedder=embedder,
        emb_tokens_start=emb_tokens_start,
        emb_cost_start=emb_cost_start,
        elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
    )


def _finish_trace(
    *,
    query_id: str,
    question: str,
    status: str,
    answer: str,
    citations: list[str],
    search_calls: int,
    inspect_calls: int,
    inspected_pages: list[int],
    context_ids: list[str],
    events: list[dict[str, Any]],
    tracker: AgentCostTracker,
    embedder: OpenAIQueryEmbedder | None,
    emb_tokens_start: int,
    emb_cost_start: float,
    elapsed_ms: float,
) -> dict[str, Any]:
    emb_tokens = (embedder.total_prompt_tokens - emb_tokens_start) if embedder else 0
    emb_cost = ((embedder.total_cost_usd - emb_cost_start) if embedder else 0.0)
    return {
        "query_id": query_id,
        "question": question,
        "status": status,
        "answer": answer,
        "citations": citations,
        "search_calls": search_calls,
        "inspect_calls": inspect_calls,
        "inspected_pages": inspected_pages,
        "final_context_ids": context_ids,
        "events": events,
        "model": tracker.model,
        "llm_calls": tracker.llm_calls,
        "llm_prompt_tokens": tracker.llm_prompt_tokens,
        "llm_completion_tokens": tracker.llm_completion_tokens,
        "embedding_tokens": emb_tokens,
        "cost_usd": tracker.cost_usd(emb_cost),
        "latency_ms": round(elapsed_ms, 2),
    }
