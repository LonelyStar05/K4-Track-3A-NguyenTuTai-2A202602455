from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import hashlib
import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    embeddings = _semantic_embeddings(sentences)
    groups: list[list[str]] = [[sentences[0]]]
    for index in range(1, len(sentences)):
        similarity = _cosine_similarity(embeddings[index - 1], embeddings[index])
        if similarity < threshold:
            groups.append([sentences[index]])
        else:
            groups[-1].append(sentences[index])

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


def _semantic_embeddings(sentences: list[str]):
    """Use a local transformer when enabled, otherwise a deterministic token vector."""
    if os.getenv("RAG_USE_TRANSFORMERS", "").lower() in {"1", "true", "yes"}:
        try:
            from sentence_transformers import SentenceTransformer

            allow_download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
            model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=not allow_download)
            return model.encode(sentences, normalize_embeddings=True)
        except Exception as exc:
            print(f"  Semantic model unavailable; using lexical vectors: {exc}")

    vocabulary = sorted({token for sentence in sentences for token in _tokens(sentence)})
    token_index = {token: index for index, token in enumerate(vocabulary)}
    vectors = []
    for sentence in sentences:
        vector = [0.0] * max(len(vocabulary), 1)
        for token in _tokens(sentence):
            vector[token_index[token]] += 1.0
        vectors.append(vector)
    return vectors


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _cosine_similarity(left, right) -> float:
    dot_product = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    return dot_product / (left_norm * right_norm + 1e-9)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    parent_texts = _pack_blocks(paragraphs, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "document"))
    source_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{source_id}_{parent_index}"
        parent_metadata = {
            **metadata,
            "chunk_type": "parent",
            "parent_id": parent_id,
            "chunk_index": parent_index,
        }
        parents.append(Chunk(text=parent_text, metadata=parent_metadata))

        child_blocks = re.split(r"(?<=[.!?])\s+|\n+", parent_text)
        for child_index, child_text in enumerate(_pack_blocks(child_blocks, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "chunk_index": child_index,
                    "parent_id": parent_id,
                },
                parent_id=parent_id,
            ))
    return parents, children


def _pack_blocks(blocks: list[str], max_size: int) -> list[str]:
    """Pack blocks without losing oversized content."""
    normalized: list[str] = []
    for raw_block in blocks:
        block = raw_block.strip()
        if not block:
            continue
        while len(block) > max_size:
            split_at = block.rfind(" ", 0, max_size + 1)
            if split_at <= 0:
                split_at = max_size
            normalized.append(block[:split_at].strip())
            block = block[split_at:].strip()
        if block:
            normalized.append(block)

    packed: list[str] = []
    current = ""
    for block in normalized:
        candidate = f"{current}\n\n{block}" if current else block
        if current and len(candidate) > max_size:
            packed.append(current)
            current = block
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^#{1,6}\s+.+$", flags=re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    if not matches:
        stripped = text.strip()
        return [Chunk(stripped, {**metadata, "section": "document", "strategy": "structure"})] if stripped else []

    chunks: list[Chunk] = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(
            text=preamble,
            metadata={**metadata, "section": "preamble", "strategy": "structure", "chunk_index": 0},
        ))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start():end].strip()
        header = match.group(0).strip()
        if section_text:
            chunks.append(Chunk(
                text=section_text,
                metadata={
                    **metadata,
                    "section": header.lstrip("#").strip(),
                    "header": header,
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            ))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
