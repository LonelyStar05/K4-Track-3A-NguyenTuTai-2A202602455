from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import math
import os, sys, time
import re
import unicodedata
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            use_transformers = os.getenv("RAG_USE_TRANSFORMERS", "").lower() in {"1", "true", "yes"}
            if use_transformers:
                try:
                    from sentence_transformers import CrossEncoder

                    allow_download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
                    self._model = CrossEncoder(
                        self.model_name,
                        local_files_only=not allow_download,
                    )
                except Exception as exc:
                    print(f"  Cross-encoder unavailable; using lexical reranker: {exc}")
            if self._model is None:
                self._model = _LexicalCrossEncoder()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        model = self._load_model()
        pairs = [
            (
                query,
                f"{document.get('text', '')}\n{document.get('metadata', {}).get('parent_text', '')}",
            )
            for document in documents
        ]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        adjusted_scores = [
            float(score) + _version_bonus(query, document)
            for score, document in zip(scores, documents)
        ]
        scored = sorted(
            zip(adjusted_scores, documents),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [
            RerankResult(
                text=document.get("text", ""),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=index + 1,
            )
            for index, (score, document) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": index, "text": document.get("text", ""), "meta": document.get("metadata", {})}
                for index, document in enumerate(documents)
            ]
            ranked = self._model.rerank(RerankRequest(query=query, passages=passages))[:top_k]
            return [
                RerankResult(
                    text=item["text"],
                    original_score=float(documents[int(item["id"])].get("score", 0.0)),
                    rerank_score=float(item["score"]),
                    metadata=dict(item.get("meta", {})),
                    rank=index + 1,
                )
                for index, item in enumerate(ranked)
            ]
        except (ImportError, RuntimeError, ValueError):
            return CrossEncoderReranker(model_name="lexical-fallback").rerank(query, documents, top_k)


class _LexicalCrossEncoder:
    """Offline cross-encoder substitute that scores token and number overlap."""

    @staticmethod
    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [_lexical_relevance(query, document) for query, document in pairs]


def _lexical_relevance(query: str, document: str) -> float:
    query_tokens = _content_tokens(query)
    document_tokens = _content_tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = query_tokens & document_tokens
    coverage = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(document_tokens)
    query_bigrams = _bigrams(query)
    document_bigrams = _bigrams(document)
    bigram_coverage = len(query_bigrams & document_bigrams) / max(len(query_bigrams), 1)
    score = 0.65 * coverage + 0.1 * precision + 0.25 * bigram_coverage

    query_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", query))
    document_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", document))
    if query_numbers:
        score += 0.15 * len(query_numbers & document_numbers) / len(query_numbers)
    return math.tanh(score * 2.0)


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "ai", "bao", "bi", "cac", "cho", "co", "cua", "duoc", "gi", "la", "moi",
        "mot", "nao", "nhan", "nhieu", "nhung", "o", "phai", "the", "thi", "trong",
        "va", "voi", "khong", "nhân", "viên", "được", "là", "có", "bao", "nhiêu",
        "một", "nếu", "từ", "và", "gì", "chiếc", "mới", "thì", "nào", "khi",
    }
    return {
        token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stopwords
    }


def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = [
        token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if token in _content_tokens(text)
    ]
    return set(zip(tokens, tokens[1:]))


def _version_bonus(query: str, document: dict) -> float:
    query_lower = query.lower()
    source = str(document.get("metadata", {}).get("source", "")).lower()
    text = f"{document.get('text', '')} {document.get('metadata', {}).get('parent_text', '')}".lower()
    asks_for_old = any(marker in query_lower for marker in ["cũ", "2022", "2023", "v1", "1.0"])
    is_old = any(marker in source for marker in ["v1", "v2023"]) or any(
        marker in text for marker in ["trạng thái: đã thay thế", "phiên bản cũ"]
    )
    is_current = any(marker in source for marker in ["v2", "v2024"]) or any(
        marker in text for marker in ["phiên bản hiện hành", "thay thế hoàn toàn"]
    )
    query_source_tokens = set(_fold_text(query).split())
    source_tokens = set(_fold_text(re.sub(r"\.(?:md|pdf)$", "", source).replace("_", " ")).split())
    source_bonus = 0.06 * min(len(query_source_tokens & source_tokens), 3)
    if asks_for_old:
        return source_bonus + (0.05 if is_old else 0.0)
    if is_old:
        return source_bonus - 0.05
    if is_current:
        return source_bonus + 0.05
    return source_bonus


def _fold_text(text: str) -> str:
    normalized = "".join(
        character for character in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return " ".join(re.findall(r"\w+", normalized))


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
