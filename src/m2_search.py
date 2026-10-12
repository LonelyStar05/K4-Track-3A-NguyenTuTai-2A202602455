from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import math
import os, sys
import re
import unicodedata
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
    except (ImportError, RuntimeError):
        segmented = text
    return " ".join(segmented.replace("_", " ").lower().split())


def _tokenize(text: str) -> list[str]:
    segmented = segment_vietnamese(text)
    folded = "".join(
        character for character in unicodedata.normalize("NFKD", segmented)
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.findall(r"\w+", folded, flags=re.UNICODE)


class _SimpleBM25:
    """Small BM25 implementation used when rank-bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        self.doc_freqs: dict[str, int] = {}
        for document in corpus:
            for token in set(document):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        document_count = len(self.corpus)
        for document in self.corpus:
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.doc_freqs.get(token, 0)
                inverse_document_frequency = math.log(
                    1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * len(document) / max(self.avgdl, 1e-9)
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1.0) / denominator
            scores.append(score)
        return scores


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[index]["text"],
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices
            if float(scores[index]) > 0.0
        ]


class DenseSearch:
    def __init__(self):
        self.client = None
        self._local_collections: dict[str, tuple[list[dict], object]] = {}
        try:
            from qdrant_client import QdrantClient

            try:
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
                self.client.get_collections()
            except Exception:
                self.client = QdrantClient(":memory:")
        except ImportError:
            self.client = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            use_transformers = os.getenv("RAG_USE_TRANSFORMERS", "").lower() in {"1", "true", "yes"}
            if use_transformers:
                try:
                    from sentence_transformers import SentenceTransformer

                    allow_download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
                    self._encoder = SentenceTransformer(
                        EMBEDDING_MODEL,
                        local_files_only=not allow_download,
                    )
                except Exception as exc:
                    print(f"  Dense model unavailable; using hashing encoder: {exc}")
            if self._encoder is None:
                self._encoder = _HashingEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        texts = [chunk.get("text", "") for chunk in chunks]
        if not texts:
            self._local_collections[collection] = ([], [])
            return
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        if self.client is None:
            self._local_collections[collection] = (list(chunks), vectors)
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        vector_size = len(vectors[0])
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        query_vector = self._get_encoder().encode(query)
        if self.client is None:
            documents, vectors = self._local_collections.get(collection, ([], []))
            ranked = sorted(
                (
                    (_cosine(query_vector, vector), document)
                    for document, vector in zip(documents, vectors)
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]
            return [
                SearchResult(
                    text=document.get("text", ""),
                    score=float(score),
                    metadata=dict(document.get("metadata", {})),
                    method="dense",
                )
                for score, document in ranked
                if score > 0.0
            ]

        vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        if hasattr(self.client, "query_points"):
            points = self.client.query_points(collection_name=collection, query=vector, limit=top_k).points
        else:  # qdrant-client 1.9 (pinned cho Qdrant v1.9.7) chỉ có search()
            points = self.client.search(collection_name=collection, query_vector=vector, limit=top_k)
        return [
            SearchResult(
                text=point.payload.get("text", ""),
                score=float(point.score),
                metadata={key: value for key, value in point.payload.items() if key != "text"},
                method="dense",
            )
            for point in points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked
    ]


class _HashingEncoder:
    """Dependency-free dense fallback based on word and character features."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, texts, show_progress_bar: bool = False):
        import numpy as np

        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        matrix = np.zeros((len(values), self.dimension), dtype="float32")
        for row, text in enumerate(values):
            normalized = segment_vietnamese(text)
            features = _tokenize(normalized)
            compact = re.sub(r"\s+", " ", normalized)
            features.extend(compact[index:index + 3] for index in range(max(len(compact) - 2, 0)))
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                column = int.from_bytes(digest, "little") % self.dimension
                matrix[row, column] += 1.0
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix[0] if single else matrix


def _cosine(left, right) -> float:
    import numpy as np

    left_array = np.asarray(left, dtype="float32")
    right_array = np.asarray(right, dtype="float32")
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        candidate_k = max(BM25_TOP_K, DENSE_TOP_K, top_k * 2)
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], top_k=candidate_k)
        diversified = []
        source_counts: dict[str, int] = {}
        for result in fused:
            source = str(result.metadata.get("source", ""))
            if source and source_counts.get(source, 0) >= 2:
                continue
            diversified.append(result)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(diversified) >= top_k:
                break
        return diversified


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
