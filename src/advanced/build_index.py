"""Full-document 124-page multimodal indexer for WEF Global Risks Report 2024."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request
from urllib.parse import urlparse

import pg8000.dbapi
import pymupdf

from ..config import load_settings
from ..embeddings import OpenAIQueryEmbedder
from ..retrievers import _vector_literal
from .download_doc import DEFAULT_PDF_PATH, download_wef_report

SOURCE_ID = "wef-grr-2024"
DEFAULT_INDEX_DIR = Path("data/advanced/index")

# Pages known to be cover/imprint/decorative-only visuals (still noted in manifest)
COVER_OR_IMPRINT_PAGES = {1, 2, 124}


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_reading_order_blocks(page: pymupdf.Page) -> list[dict[str, Any]]:
    """Extract text blocks sorted cleanly in page reading order (handles 2-col and 3-col layouts)."""
    width = float(page.rect.width)
    mid_x = width * 0.5
    raw_blocks = page.get_text("blocks")
    text_blocks: list[dict[str, Any]] = []
    for blk in raw_blocks:
        if len(blk) < 7 or blk[6] != 0:
            continue
        x0, y0, x1, y1, txt = float(blk[0]), float(blk[1]), float(blk[2]), float(blk[3]), _clean_text(str(blk[4]))
        if not txt:
            continue
        # Classify span: wide header/full-width vs left column vs right column
        if (x1 - x0) > width * 0.58 or (x0 < mid_x - 30 and x1 > mid_x + 30):
            col = 0 if y0 < 150 else 1
            band = 0 if y0 < 150 else 2
        else:
            band = 1
            col = 1 if x1 <= mid_x + 25 else 2
        text_blocks.append(
            {"bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "text": txt, "band": band, "col": col}
        )
    text_blocks.sort(key=lambda b: (b["band"], b["col"], b["bbox"][1], b["bbox"][0]))
    return text_blocks


def _chunk_blocks(
    page_num: int,
    blocks: list[dict[str, Any]],
    max_chars: int = 950,
    overlap_chars: int = 120,
) -> list[dict[str, Any]]:
    """Group ordered blocks on a page into overlapping semantic chunks preserving bounding boxes."""
    if not blocks:
        return []
    chunks: list[dict[str, Any]] = []
    cur_texts: list[str] = []
    cur_bboxes: list[list[float]] = []
    cur_len = 0

    def flush() -> None:
        if not cur_texts:
            return
        idx = len(chunks) + 1
        x0 = min(b[0] for b in cur_bboxes)
        y0 = min(b[1] for b in cur_bboxes)
        x1 = max(b[2] for b in cur_bboxes)
        y1 = max(b[3] for b in cur_bboxes)
        joined = "\n\n".join(cur_texts).strip()
        if joined:
            chunks.append(
                {
                    "id": f"{SOURCE_ID}:p{page_num:03d}:text:{idx:02d}",
                    "source_id": SOURCE_ID,
                    "source": "WEF Global Risks Report 2024",
                    "page": page_num,
                    "modality": "text",
                    "figure_id": None,
                    "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    "crop_path": None,
                    "page_image_path": f"data/advanced/index/pages/p{page_num:03d}.png",
                    "text": joined,
                    "metadata": {"page": page_num, "modality": "text"},
                }
            )

    for blk in blocks:
        txt = blk["text"]
        if cur_texts and cur_len + len(txt) + 2 > max_chars:
            flush()
            tail_txt = cur_texts[-1]
            if len(tail_txt) <= overlap_chars:
                cur_texts = [tail_txt, txt]
                cur_bboxes = [cur_bboxes[-1], blk["bbox"]]
                cur_len = len(tail_txt) + len(txt) + 2
            else:
                cur_texts = [txt]
                cur_bboxes = [blk["bbox"]]
                cur_len = len(txt)
        else:
            cur_texts.append(txt)
            cur_bboxes.append(blk["bbox"])
            cur_len += len(txt) + 2
    flush()
    return chunks


def _extract_tables_for_page(page: pymupdf.Page, page_num: int) -> list[dict[str, Any]]:
    """Extract structured tables via PyMuPDF find_tables plus layout-aware Appendix C columns."""
    out: list[dict[str, Any]] = []
    seen_sigs: set[str] = set()

    try:
        found = page.find_tables().tables
    except Exception:
        found = []

    for idx, tab in enumerate(found, 1):
        rows = tab.extract() or []
        clean_rows = [[_clean_text(str(cell or "")).replace("\n", " ") for cell in r] for r in rows]
        clean_rows = [r for r in clean_rows if any(c for c in r)]
        if len(clean_rows) < 2:
            continue
        lines = [" | ".join(r) for r in clean_rows]
        text = f"[Table p{page_num} #{idx}]\n" + "\n".join(lines)
        sig = text[:160]
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        bbox = [round(float(v), 1) for v in tab.bbox]
        out.append(
            {
                "id": f"{SOURCE_ID}:p{page_num:03d}:table:{len(out)+1:02d}",
                "source_id": SOURCE_ID,
                "source": "WEF Global Risks Report 2024",
                "page": page_num,
                "modality": "table",
                "figure_id": f"Table p{page_num} #{idx}",
                "bbox": bbox,
                "crop_path": f"data/advanced/index/crops/p{page_num:03d}_visual.png",
                "page_image_path": f"data/advanced/index/pages/p{page_num:03d}.png",
                "text": text,
                "metadata": {"page": page_num, "modality": "table", "rows": len(clean_rows)},
            }
        )

    # Appendix C (pages 103-111): 3-column country top-5 risks layout not detected as bordered tables
    if 103 <= page_num <= 111:
        w = float(page.rect.width)
        col_w = w / 3.0
        col_blocks: dict[int, list[tuple[float, str]]] = {0: [], 1: [], 2: []}
        for blk in page.get_text("blocks"):
            if len(blk) < 7 or blk[6] != 0:
                continue
            x0, y0, _, y1, raw = float(blk[0]), float(blk[1]), float(blk[2]), float(blk[3]), _clean_text(str(blk[4]))
            if not raw or y0 < 160 or y1 > 790:
                continue
            if raw.startswith(("Risk categories", "Global Risks Report")):
                continue
            col_idx = min(2, max(0, int(x0 // col_w)))
            col_blocks[col_idx].append((y0, raw.replace("\n", " ")))
        col_parts: list[str] = []
        for col_idx in (0, 1, 2):
            items = [t for _, t in sorted(col_blocks[col_idx])]
            if items:
                col_parts.append(f"Column {col_idx + 1}: " + " | ".join(items))
        if col_parts:
            text = f"[Table C.2 Country Risk Perceptions p{page_num}]\n" + "\n".join(col_parts)
            out.append(
                {
                    "id": f"{SOURCE_ID}:p{page_num:03d}:table:{len(out)+1:02d}",
                    "source_id": SOURCE_ID,
                    "source": "WEF Global Risks Report 2024",
                    "page": page_num,
                    "modality": "table",
                    "figure_id": f"TABLE C.2 (p{page_num})",
                    "bbox": [24.0, 150.0, round(w - 24.0, 1), 800.0],
                    "crop_path": f"data/advanced/index/crops/p{page_num:03d}_visual.png",
                    "page_image_path": f"data/advanced/index/pages/p{page_num:03d}.png",
                    "text": text,
                    "metadata": {"page": page_num, "modality": "table", "table_id": "C.2"},
                }
            )
    return out


def _detect_visual_info(page: pymupdf.Page, page_num: int) -> dict[str, Any]:
    """Identify whether a page contains informative figures, charts, diagrams, maps, or visual tables."""
    txt = page.get_text()
    compact_lines = [re.sub(r"\s+", "", l.strip()) for l in txt.splitlines() if l.strip()]
    fig_labels = [
        l for l in compact_lines if re.match(r"^(FIGURE|TABLE|BOX)[A-Z0-9.]+$", l, re.I)
    ]
    n_drawings = len(page.get_drawings())
    imgs = page.get_images(full=True)
    # Check non-decorative raster images: either multi-image figure (e.g. p47 maps) or accompanied by FIGURE/BOX
    has_non_trivial_img = any(im[2] >= 250 and im[3] >= 250 for im in imgs)
    is_table_appendix = 95 <= page_num <= 114
    is_cover_or_blank = page_num in COVER_OR_IMPRINT_PAGES or len(txt.strip()) < 40

    if is_cover_or_blank:
        return {"has_visual": False, "reason": "cover_or_imprint", "labels": fig_labels}

    # Pages with ONLY a decorative Unsplash stock photo (no FIGURE/BOX and low drawings)
    if has_non_trivial_img and not fig_labels and n_drawings < 25 and "Unsplash" in txt:
        return {"has_visual": False, "reason": "decorative_stock_photo_only", "labels": []}

    has_visual = bool(fig_labels) or n_drawings >= 25 or is_table_appendix or len(imgs) >= 4
    if not has_visual:
        return {"has_visual": False, "reason": "text_only_page", "labels": []}

    # Estimate visual crop bbox using drawings + figure header/images/tables
    w, h = float(page.rect.width), float(page.rect.height)
    rects: list[pymupdf.Rect] = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r and r.width < w * 0.98 and r.height < h * 0.95 and r.height > 8:
            rects.append(r)
    for info in page.get_image_info(xrefs=False):
        r = pymupdf.Rect(info["bbox"])
        if r.width > 40 and r.height > 40:
            rects.append(r)
    if rects:
        x0 = max(12.0, min(r.x0 for r in rects) - 18.0)
        y0 = max(24.0, min(r.y0 for r in rects) - 45.0)
        x1 = min(w - 12.0, max(r.x1 for r in rects) + 18.0)
        y1 = min(h - 24.0, max(r.y1 for r in rects) + 35.0)
        if (x1 - x0) < 160 or (y1 - y0) < 120:
            bbox = [16.0, 40.0, round(w - 16.0, 1), round(h - 32.0, 1)]
        else:
            bbox = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
    else:
        bbox = [16.0, 40.0, round(w - 16.0, 1), round(h - 32.0, 1)]

    return {
        "has_visual": True,
        "reason": "figure_chart_map_or_table",
        "labels": fig_labels,
        "bbox": bbox,
        "n_drawings": n_drawings,
        "n_images": len(imgs),
    }


def _describe_visual_page_with_vision(
    page_num: int,
    crop_path: Path,
    labels: list[str],
    api_key: str,
    model: str = "gpt-6-luna",
) -> dict[str, Any]:
    """Call gpt-6-luna vision on the cropped visual image to build a searchable visual index entry."""
    b64 = base64.b64encode(crop_path.read_bytes()).decode("ascii")
    prompt = (
        f"You are indexing page {page_num} of the WEF Global Risks Report 2024 (labels detected: {labels}). "
        "Summarize ALL informational visual elements shown (chart, table, diagram/network, map, or box). "
        "Include: (1) Modality (chart/diagram/map/table/box), (2) Exact Figure/Table title, "
        "(3) Axes, units, legend categories/colors, node/edge rules if diagram, "
        "(4) Exact visible headings, numbers, percentages, rankings, countries, or values so text search finds this page. "
        "Skip decorative photos. Be concise, accurate, and factual."
    )
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64, "detail": "high"},
                    },
                ],
            }
        ],
    }
    if model.startswith(("gpt-6", "gpt-5")):
        payload["reasoning_effort"] = "none"

    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(1, 4):
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            usage = body.get("usage") or {}
            return {
                "page": page_num,
                "summary": (body["choices"][0]["message"]["content"] or "").strip(),
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"Vision index p{page_num} HTTP {exc.code}: {err_body}")
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise last_err from exc
        except Exception as exc:
            last_err = exc
            if attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise RuntimeError(f"Vision index p{page_num} failed: {exc}") from exc
    raise RuntimeError(f"Vision index p{page_num} failed: {last_err}")


def _embed_all_chunks(
    chunks: list[dict[str, Any]],
    api_key: str,
    model_name: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> tuple[list[list[float]], int, float]:
    """Batch-embed all chunks via OpenAI Embeddings API."""
    vectors: list[list[float]] = []
    total_tokens = 0
    endpoint = "https://api.openai.com/v1/embeddings"

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c["text"][:8000] for c in batch]
        payload = json.dumps({"model": model_name, "input": texts}).encode("utf-8")
        for attempt in range(1, 4):
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                usage = body.get("usage") or {}
                total_tokens += int(usage.get("prompt_tokens", usage.get("total_tokens", 0)))
                data_items = sorted(body["data"], key=lambda x: x["index"])
                vectors.extend(list(item["embedding"]) for item in data_items)
                break
            except Exception as exc:
                if attempt < 3:
                    time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(f"Batch embedding failed at {start}: {exc}") from exc
    cost_usd = round((total_tokens / 1_000_000.0) * 0.02, 8)
    return vectors, total_tokens, cost_usd


def _upsert_pgvector(
    database_url: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> None:
    """Persist all multimodal chunks and 1536d embeddings into PostgreSQL table advanced_chunks."""
    parsed = urlparse(database_url)
    conn = pg8000.dbapi.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 5432,
        database=(parsed.path or "/").lstrip("/"),
        timeout=15,
    )
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS advanced_chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                page INTEGER NOT NULL,
                modality TEXT NOT NULL,
                figure_id TEXT,
                bbox JSONB,
                crop_path TEXT,
                page_image_path TEXT,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector(1536) NOT NULL
            )
            """
        )
        cur.execute("TRUNCATE TABLE advanced_chunks")
        for chunk, vec in zip(chunks, vectors):
            cur.execute(
                """
                INSERT INTO advanced_chunks
                    (chunk_id, source_id, page, modality, figure_id, bbox, crop_path, page_image_path, text, metadata, embedding)
                VALUES
                    (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::vector)
                """,
                (
                    chunk["id"],
                    chunk["source_id"],
                    int(chunk["page"]),
                    chunk["modality"],
                    chunk.get("figure_id"),
                    json.dumps(chunk.get("bbox")),
                    chunk.get("crop_path"),
                    chunk.get("page_image_path"),
                    chunk["text"],
                    json.dumps(chunk.get("metadata") or {}),
                    _vector_literal(vec),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def build_advanced_index(
    pdf_path: Path = DEFAULT_PDF_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    vision_model: str = "gpt-6-luna",
    max_workers: int = 8,
) -> dict[str, Any]:
    """Run full-document 124-page parsing, cropping, vision indexing, embedding, and pgvector upsert."""
    t0 = time.perf_counter()
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is missing in .env")

    source_meta = download_wef_report(pdf_path)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    pages_dir = index_dir / "pages"
    crops_dir = index_dir / "crops"
    pages_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    cache_path = index_dir / "vision_cache.json"
    vision_cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )

    all_chunks: list[dict[str, Any]] = []
    visual_jobs: list[dict[str, Any]] = []
    skipped_visual_pages: list[dict[str, Any]] = []
    pages_without_text: list[int] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Render full page PNG for universal live visual verification
        page_png = pages_dir / f"p{page_num:03d}.png"
        if not page_png.exists():
            page.get_pixmap(dpi=140).save(page_png)

        blocks = _extract_reading_order_blocks(page)
        if not blocks:
            pages_without_text.append(page_num)
        text_chunks = _chunk_blocks(page_num, blocks)
        all_chunks.extend(text_chunks)

        table_chunks = _extract_tables_for_page(page, page_num)
        all_chunks.extend(table_chunks)

        vis_info = _detect_visual_info(page, page_num)
        if vis_info["has_visual"]:
            bbox = vis_info["bbox"]
            crop_png = crops_dir / f"p{page_num:03d}_visual.png"
            if not crop_png.exists():
                clip = pymupdf.Rect(*bbox)
                page.get_pixmap(dpi=180, clip=clip).save(crop_png)
            visual_jobs.append(
                {
                    "page": page_num,
                    "crop_path": crop_png,
                    "rel_crop_path": f"data/advanced/index/crops/p{page_num:03d}_visual.png",
                    "bbox": bbox,
                    "labels": vis_info["labels"],
                }
            )
        else:
            skipped_visual_pages.append({"page": page_num, "reason": vis_info["reason"]})

    doc.close()

    # Run gpt-6-luna vision indexing for uncached visual pages
    pending = [j for j in visual_jobs if str(j["page"]) not in vision_cache]
    vision_prompt_tokens = 0
    vision_completion_tokens = 0
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(
                    _describe_visual_page_with_vision,
                    job["page"],
                    job["crop_path"],
                    job["labels"],
                    settings.openai_api_key,
                    vision_model,
                ): job
                for job in pending
            }
            for fut in as_completed(futs):
                res = fut.result()
                vision_cache[str(res["page"])] = res
        cache_path.write_text(json.dumps(vision_cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for job in visual_jobs:
        rec = vision_cache[str(job["page"])]
        vision_prompt_tokens += int(rec.get("prompt_tokens", 0))
        vision_completion_tokens += int(rec.get("completion_tokens", 0))
        fig_id = ", ".join(job["labels"]) if job["labels"] else f"Visual p{job['page']}"
        all_chunks.append(
            {
                "id": f"{SOURCE_ID}:p{job['page']:03d}:visual:01",
                "source_id": SOURCE_ID,
                "source": "WEF Global Risks Report 2024",
                "page": job["page"],
                "modality": "visual",
                "figure_id": fig_id,
                "bbox": job["bbox"],
                "crop_path": job["rel_crop_path"],
                "page_image_path": f"data/advanced/index/pages/p{job['page']:03d}.png",
                "text": f"[Visual Index p{job['page']} | {fig_id}]\n{rec['summary']}",
                "metadata": {"page": job["page"], "modality": "visual", "labels": job["labels"]},
            }
        )

    vectors, emb_tokens, emb_cost = _embed_all_chunks(
        all_chunks, settings.openai_api_key, model_name=settings.embedding_model
    )
    _upsert_pgvector(settings.database_url, all_chunks, vectors)

    (index_dir / "chunks.json").write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    vision_cost = round(
        (vision_prompt_tokens / 1_000_000.0) * 0.10 + (vision_completion_tokens / 1_000_000.0) * 0.50,
        6,
    )
    manifest = {
        "source": source_meta,
        "total_pages": total_pages,
        "total_chunks": len(all_chunks),
        "chunks_by_modality": {
            "text": sum(1 for c in all_chunks if c["modality"] == "text"),
            "table": sum(1 for c in all_chunks if c["modality"] == "table"),
            "visual": sum(1 for c in all_chunks if c["modality"] == "visual"),
        },
        "visual_pages_indexed": len(visual_jobs),
        "pages_without_text": pages_without_text,
        "skipped_visual_pages": skipped_visual_pages,
        "limitations": (
            "Pages 1, 2, and 124 are cover/imprint art; pages flagged decorative_stock_photo_only "
            "contain Unsplash photography without data labels and are excluded from visual chunks "
            "(all pages remain rendered in pages/pXXX.png). Vision summaries may contain minor "
            "rounding; live inspect_page_image tool reads original high-res PNGs at query time."
        ),
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": 1536,
        "embedding_tokens": emb_tokens,
        "embedding_cost_usd": emb_cost,
        "vision_indexing_model": vision_model,
        "vision_prompt_tokens": vision_prompt_tokens,
        "vision_completion_tokens": vision_completion_tokens,
        "vision_indexing_cost_usd": vision_cost,
        "total_indexing_cost_usd": round(emb_cost + vision_cost, 6),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "pgvector_table": "advanced_chunks",
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 124-page multimodal index for WEF 2024.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--vision-model", default="gpt-6-luna")
    args = parser.parse_args()
    manifest = build_advanced_index(args.pdf, args.index_dir, vision_model=args.vision_model)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
