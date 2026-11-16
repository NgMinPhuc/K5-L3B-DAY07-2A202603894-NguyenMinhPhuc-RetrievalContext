"""Index a long PDF (text + tables + charts + figures) for the advanced challenge.

Pipeline per page:
  1. Render the page to PNG with poppler pdftoppm (kept for verification and visual citations).
  2. Extract the text layer with pypdf and split it into overlapping chunks.
  3. Send the page image (+ its text layer as a hint) to a vision model that
     transcribes every informative table, chart, figure/diagram/map into a JSON
     list of elements. Each element becomes one "visual" chunk tied to the page
     and to a page region (caption bbox located in the text layer when possible).
  4. Embed all chunks with text-embedding-3-small and store vectors locally.

Vision output is cached per page so reruns only redo failed pages. No benchmark
file is read here: gold answers/regions never enter the index.

Run from the repo root:
  .venv/bin/python -m advanced.ingest --pdf advanced/source/<file>.pdf
"""
import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "advanced/index"

VISION_PROMPT = """You are indexing one page of a report for a retrieval system.
Return JSON: {"elements": [...]} listing EVERY informative visual element on the page:
tables, charts/graphs, figures, diagrams, maps, infographics, boxes with data.
Skip purely decorative images, logos, photos without information, page headers/footers.
For each element return:
- "type": one of "table", "chart", "figure", "diagram", "map", "infographic"
- "label": the printed label such as "Figure 2.3" / "Table 1.1" / "Box 3" or "" if none
- "title": the printed title/caption (verbatim)
- "region": approximate position on the page, e.g. "top half", "bottom-left", "full page"
- "content": a faithful, complete transcription:
   * table: markdown table with ALL rows/columns, header units, footnotes, source line
   * chart: chart kind; x-axis and y-axis titles/units; legend series; for EVERY series
     list the values you can read for each category/year (prefix estimates read from
     bar heights/line positions with "~"; exact printed data labels without "~");
     state the visible trend and min/max; include notes and source line
   * figure/diagram/map/infographic: what it shows, all text labels, numbers, arrows/flows,
     categories, legends, and the main message
- "notes": anything unreadable or uncertain
Use the page's text layer (given below) only to spell labels/numbers correctly; the image
is authoritative for layout. Do not invent values. If there are no informative visuals,
return {"elements": []}."""


def _chunk_text(text: str, size: int = 1400, overlap: int = 200) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = (chunks[-1][-overlap:] + "\n\n" + p) if chunks and overlap else p
    if cur:
        chunks.append(cur)
    return chunks


def _page_words(pdf: Path, page_no: int) -> list[tuple[str, list[float]]]:
    """Word boxes for one page via poppler `pdftotext -bbox` (PDF points)."""
    import html
    import subprocess
    out = subprocess.run(["pdftotext", "-bbox", "-f", str(page_no), "-l", str(page_no), str(pdf), "-"],
                         capture_output=True, text=True, check=True).stdout
    words = []
    for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', out):
        words.append((html.unescape(m.group(5)), [round(float(m.group(i)), 1) for i in range(1, 5)]))
    return words


def _caption_bbox(words: list[tuple[str, list[float]]], label: str, title: str) -> list[float] | None:
    """Locate the caption (label, else first title words) on the page; return its bbox."""
    for needle in (label, " ".join(title.split()[:5])):
        toks = needle.split()
        if len(" ".join(toks)) < 4:
            continue
        n = len(toks)
        for i in range(len(words) - n + 1):
            if [w.rstrip(":.") for w, _ in words[i:i + n]] == [t.rstrip(":.") for t in toks]:
                boxes = [b for _, b in words[i:i + n]]
                return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return None


def _vision_page(client: Any, model: str, png: Path, text_layer: str, retries: int = 3) -> dict[str, Any]:
    b64 = base64.b64encode(png.read_bytes()).decode()
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                max_completion_tokens=6000,
                messages=[
                    {"role": "system", "content": VISION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Text layer of this page (may be incomplete):\n" + text_layer[:6000]},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    ]},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            if not isinstance(data.get("elements"), list):
                raise ValueError("missing elements list")
            return {"elements": data["elements"],
                    "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                              "completion_tokens": resp.usage.completion_tokens}}
        except Exception as exc:  # retried, then surfaced in the manifest
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * attempt)
    return {"elements": [], "error": last}


_ANNEX = re.compile(r"^\s*Phụ lục\s+(\d+)\s*[.:]\s*(.*)")
_SUBSEC = re.compile(r"^\s*([A-H])\.\s+(\S.{3,})")
_LABEL = re.compile(r"^(Bảng|Hình|Hộp)\s+\d+")


def _enrich(chunks: list[dict[str, Any]], page_texts: dict[int, str]) -> None:
    """Add document-structure context derived from the PDF itself (never from gold):
    - section header for annex pages ("Phụ lục 3 – mục B ...") from page headings;
    - label continuation for tables that continue from the previous page.
    The header is prepended to chunk text so lexical/dense retrieval can match labels."""
    section: dict[int, str] = {}
    annex, sub = "", ""
    for p in sorted(page_texts):
        lines = [l for l in page_texts[p].splitlines() if l.strip()]
        head = lines[:3]
        is_toc = sum(bool(_ANNEX.match(l)) for l in lines) >= 2  # table of contents lists annexes
        for l in ([] if is_toc else head):
            m = _ANNEX.match(l)
            if m:
                annex, sub = f"Phụ lục {m.group(1)}. {m.group(2).strip()}", ""
                break
        if annex:
            for l in lines:
                m = _SUBSEC.match(l)
                if m:
                    sub = f"mục {m.group(1)}. {m.group(2).strip()[:80]}"
                    # e.g. "Phụ lục 3B" alias for references like "Phụ lục 3B"
                    break
        if annex:
            num = re.match(r"Phụ lục (\d+)", annex).group(1)
            alias = f" (Phụ lục {num}{sub[4]})" if sub else ""
            section[p] = f"{annex}{' – ' + sub if sub else ''}{alias}"
    last_table_label: dict[int, str] = {}
    for c in chunks:
        md = c["metadata"]
        if md.get("modality") == "visual" and md.get("visual_type") == "table" and _LABEL.match(md.get("label") or ""):
            last_table_label[c["page"]] = md["label"].rstrip(".")
    for c in chunks:
        md, p = c["metadata"], c["page"]
        if md.get("modality") == "visual" and md.get("visual_type") == "table" and not _LABEL.match(md.get("label") or ""):
            prev = last_table_label.get(p - 1)
            if prev and str(md.get("region", "")).lower().startswith(("top", "upper", "full")):
                md["label"] = f"{prev} (tiếp)"
                md["label_source"] = "continued from previous page"
            elif section.get(p):
                md["label"] = section[p].split(".")[0]
                md["label_source"] = "annex heading"
        ctx = []
        if section.get(p):
            ctx.append(section[p])
            md["section"] = section[p]
        if md.get("modality") == "visual" and md.get("label_source"):
            ctx.append(md["label"])
        if ctx:
            c["text"] = "[" + " | ".join(ctx) + f" | trang {p}]\n" + c["text"]


def build(pdf: Path, out: Path, *, dpi: int = 130, workers: int = 8, source_id: str | None = None,
          skip_pages: set[int] | None = None) -> dict[str, Any]:
    import subprocess
    from dotenv import load_dotenv
    from pypdf import PdfReader
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    model = os.environ.get("VISION_MODEL") or os.environ["OPENAI_CHAT_MODEL"]
    client = OpenAI()
    source_id = source_id or pdf.stem
    pages_dir, cache_dir = out / "pages", out / "vision_cache"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf))
    n_pages = len(reader.pages)
    skip_pages = skip_pages or set()
    started = time.perf_counter()

    page_texts: dict[int, str] = {}
    for i, page in enumerate(reader.pages, 1):
        page_texts[i] = page.extract_text() or ""
    if not all((pages_dir / f"p{i:03d}.png").exists() for i in page_texts):
        # pdftoppm names files <prefix>-<n>.png with zero padding by page count
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(pages_dir / "raw")], check=True)
        for f in pages_dir.glob("raw-*.png"):
            f.rename(pages_dir / f"p{int(f.stem.split('-')[-1]):03d}.png")

    todo = [i for i in page_texts if i not in skip_pages and not (cache_dir / f"p{i:03d}.json").exists()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_vision_page, client, model, pages_dir / f"p{i:03d}.png", page_texts[i]): i for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            (cache_dir / f"p{i:03d}.json").write_text(json.dumps(fut.result(), ensure_ascii=False, indent=1), encoding="utf-8")
            if n % 10 == 0:
                print(f"vision {n}/{len(todo)}", flush=True)

    chunks: list[dict[str, Any]] = []
    vision_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    page_stats, failed = [], []
    for i in range(1, n_pages + 1):
        words = _page_words(pdf, i)
        text_chunks = _chunk_text(page_texts[i])
        for k, t in enumerate(text_chunks):
            chunks.append({"id": f"p{i:03d}-t{k}", "text": t, "source_id": source_id, "source": pdf.name,
                           "page": i, "metadata": {"modality": "text", "region": "page text"}})
        visual = {"elements": []} if i in skip_pages else json.loads((cache_dir / f"p{i:03d}.json").read_text(encoding="utf-8"))
        if visual.get("error"):
            failed.append({"page": i, "error": visual["error"]})
        for key in vision_usage:
            vision_usage[key] += (visual.get("usage") or {}).get(key, 0)
        n_vis = 0
        for k, el in enumerate(visual.get("elements") or []):
            if not isinstance(el, dict) or not str(el.get("content", "")).strip():
                continue
            etype = str(el.get("type") or "figure")
            label, title = str(el.get("label") or ""), str(el.get("title") or "")
            bbox = _caption_bbox(words, label, title)
            header = f"[{etype.upper()}] {label} {title}".strip()
            body = f"{header}\n(page {i}, region: {el.get('region', '')})\n{el['content']}"
            if el.get("notes"):
                body += f"\nNotes: {el['notes']}"
            chunks.append({"id": f"p{i:03d}-v{k}", "text": body, "source_id": source_id, "source": pdf.name,
                           "page": i, "metadata": {"modality": "visual", "visual_type": etype, "label": label,
                                                   "title": title, "region": el.get("region", ""),
                                                   "caption_bbox": bbox, "page_image": f"pages/p{i:03d}.png"}})
            n_vis += 1
        page_stats.append({"page": i, "text_chars": len(page_texts[i].strip()), "text_chunks": len(text_chunks),
                           "visual_elements": n_vis, "skipped": i in skip_pages})

    _enrich(chunks, page_texts)

    from starter.day6.src.embeddings import OpenAIEmbedder
    import numpy as np
    embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
    vectors = np.asarray(embedder.embed_documents([c["text"] for c in chunks]), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(out / "embeddings.npy", vectors)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    manifest = {
        "source_pdf": str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf),
        "source_id": source_id, "pages": n_pages, "render_dpi": dpi, "vision_model": model,
        "embedding_model": "text-embedding-3-small", "embedding_dim": int(vectors.shape[1]),
        "chunks_total": len(chunks),
        "chunks_text": sum(c["metadata"]["modality"] == "text" for c in chunks),
        "chunks_visual": sum(c["metadata"]["modality"] == "visual" for c in chunks),
        "visual_by_type": {t: sum(c["metadata"].get("visual_type") == t for c in chunks)
                           for t in sorted({c["metadata"].get("visual_type") for c in chunks if c["metadata"]["modality"] == "visual"})},
        "pages_without_text_layer": [s["page"] for s in page_stats if s["text_chars"] < 20],
        "skipped_pages": sorted(skip_pages), "vision_failed_pages": failed,
        "vision_usage": vision_usage, "embedding_tokens": embedder.total_tokens,
        "build_seconds": round(time.perf_counter() - started, 1),
        "page_stats": page_stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--source-id")
    p.add_argument("--skip-pages", default="", help="comma list of pages not sent to vision (e.g. blank)")
    a = p.parse_args()
    skip = {int(x) for x in a.skip_pages.split(",") if x.strip()}
    m = build(a.pdf, a.out, dpi=a.dpi, workers=a.workers, source_id=a.source_id, skip_pages=skip)
    print(json.dumps({k: v for k, v in m.items() if k != "page_stats"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
