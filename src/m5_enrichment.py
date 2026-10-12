from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, sys
import re
from concurrent.futures import ThreadPoolExecutor
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_provider import chat_completion, has_api_key


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    response = _chat(
        "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        text,
        max_tokens=150,
    )
    if response:
        return response.strip()
    return _local_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0 or not text.strip():
        return []
    response = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu hỏi một dòng.",
        text,
        max_tokens=200,
    )
    if response:
        questions = [_clean_list_item(line) for line in response.splitlines() if line.strip()]
        return [question if question.endswith("?") else f"{question}?" for question in questions[:n_questions]]

    return _local_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text:
        return text
    context = _chat(
        "Viết một câu ngắn mô tả vị trí/chủ đề của đoạn văn trong tài liệu. Chỉ trả về một câu.",
        f"Tài liệu: {document_title or 'không rõ'}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if not context:
        topic = _infer_topic(text)
        source = document_title or "tài liệu nội bộ"
        context = f"Trích từ {source}, nội dung thuộc chủ đề {topic}."
    return f"{context.strip()}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    response = _chat(
        'Trích xuất metadata và chỉ trả về JSON: {"topic":"...","entities":["..."],"category":"policy|hr|it|finance","language":"vi|en"}',
        text,
        max_tokens=150,
    )
    parsed = _parse_json(response) if response else None
    if isinstance(parsed, dict):
        return parsed
    return _local_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    system_prompt = """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "một câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""
    response = _chat(system_prompt, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", max_tokens=400)
    parsed = _parse_json(response) if response else None
    if isinstance(parsed, dict):
        return parsed

    metadata = _local_metadata(text)
    topic = metadata.get("topic", "chính sách nội bộ")
    return {
        "summary": _local_summary(text),
        "questions": _local_questions(text, 3),
        "context": f"Trích từ {source or 'tài liệu nội bộ'}, nội dung thuộc chủ đề {topic}.",
        "metadata": metadata,
    }


def _chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    use_llm_enrichment = os.getenv("USE_LLM_ENRICHMENT", "1").lower() in {"1", "true", "yes"}
    if not has_api_key() or not use_llm_enrichment:
        return ""
    try:
        return chat_completion(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=0,
        )
    except Exception as exc:
        print(f"  LLM enrichment failed; using local fallback: {exc}")
        return ""


def _parse_json(value: str):
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def _clean_list_item(value: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", value).strip()


def _local_summary(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
    summary = " ".join(sentences[:2])
    return summary or text.strip()


def _local_questions(text: str, n_questions: int) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"[.!?\n]+", text) if len(sentence.strip()) > 10]
    questions = []
    for sentence in sentences[:n_questions]:
        key_phrase = " ".join(sentence.split()[:12]).rstrip(".,;:")
        questions.append(f"Thông tin nào được quy định về {key_phrase}?")
    return questions


def _local_metadata(text: str) -> dict:
    return {
        "topic": _infer_topic(text),
        "entities": _extract_entities(text),
        "category": _infer_category(text),
        "language": "vi" if re.search(r"[À-ỹ]", text) else "en",
    }


def _infer_topic(text: str) -> str:
    tokens = [
        token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if len(token) > 3
    ]
    stopwords = {"nhân", "viên", "được", "trong", "theo", "phải", "không", "công", "việc"}
    keywords = [token for token in tokens if token not in stopwords]
    return " ".join(keywords[:4]) or "chính sách nội bộ"


def _infer_category(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["mật khẩu", "vpn", "bảo mật", "dữ liệu", "mfa"]):
        return "it"
    if any(word in lowered for word in ["lương", "chi phí", "thưởng", "tạm ứng", "hoàn chi"]):
        return "finance"
    if any(word in lowered for word in ["nghỉ", "nhân viên", "thử việc", "đào tạo", "hiệu suất"]):
        return "hr"
    return "policy"


def _extract_entities(text: str) -> list[str]:
    entities = re.findall(r"\b(?:[A-ZĐ][\wÀ-ỹ-]+(?:\s+[A-ZĐ][\wÀ-ỹ-]+){0,3}|\d+(?:[.,]\d+)?%?)\b", text)
    return list(dict.fromkeys(entity.strip() for entity in entities))[:10]


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    worker_count = int(os.getenv("ENRICHMENT_WORKERS", "2"))
    if use_combined and has_api_key() and len(chunks) > 1 and worker_count > 1:
        enriched = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(lambda chunk: _enrich_chunk(chunk, methods, use_combined), chunks)
            for i, item in enumerate(iterator):
                enriched.append(item)
                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
    else:
        enriched = [_enrich_chunk(chunk, methods, use_combined) for chunk in chunks]
        for i in range(len(enriched)):
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


def _enrich_chunk(chunk: dict, methods: list[str], use_combined: bool) -> EnrichedChunk:
    text = chunk["text"]
    source = chunk.get("metadata", {}).get("source", "")
    if use_combined:
        result = _enrich_single_call(text, source)
        summary = result.get("summary", "")
        questions = result.get("questions", [])
        context_line = result.get("context", "")
        enrichment_parts = [context_line, summary]
        if questions:
            enrichment_parts.append("Câu hỏi liên quan: " + " ".join(questions))
        enrichment_parts.append(text)
        enriched_text = "\n\n".join(part for part in enrichment_parts if part)
        auto_meta = result.get("metadata", {})
    else:
        summary = summarize_chunk(text) if "summary" in methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
        enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
        auto_meta = extract_metadata(text) if "metadata" in methods else {}

    return EnrichedChunk(
        original_text=text,
        enriched_text=enriched_text,
        summary=summary,
        hypothesis_questions=questions,
        auto_metadata={**chunk.get("metadata", {}), **auto_meta},
        method="+".join(methods),
    )


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
