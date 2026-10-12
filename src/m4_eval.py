from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import hashlib
import math
import os, sys, json
import re
from copy import deepcopy
from statistics import mean
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from src.llm_provider import get_ragas_models, has_api_key, provider_name, ragas_model_ids


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts, and ground_truths must have equal lengths")
    if not questions:
        return _aggregate([], backend="empty", evaluator_models=[])

    if not has_api_key():
        if os.getenv("RAG_ALLOW_OFFLINE_EVAL", "").lower() in {"1", "true", "yes"}:
            per_question = [
                _offline_evaluate(question, answer, context, ground_truth)
                for question, answer, context, ground_truth in zip(
                    questions, answers, contexts, ground_truths
                )
            ]
            return _aggregate(per_question, backend="offline_lexical_proxy",
                              evaluator_models=["offline-lexical-proxy"])
        raise RuntimeError("RAGAS cần API key; không lặng lẽ thay bằng proxy offline.")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    llm, embeddings = get_ragas_models()
    answer_relevancy_metric = deepcopy(answer_relevancy)
    answer_relevancy_metric.strictness = 1
    metrics = [_faithfulness_metric(), answer_relevancy_metric, context_precision, context_recall]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    metric_by_name = dict(zip(metric_names, metrics))

    def run(indices: list[int], names: list[str]) -> list[dict]:
        dataset = Dataset.from_dict({
            "question": [questions[i] for i in indices],
            "answer": [answers[i] for i in indices],
            "contexts": [contexts[i] for i in indices],
            "ground_truth": [ground_truths[i] for i in indices],
        })
        result = evaluate(
            dataset, metrics=[metric_by_name[n] for n in names], llm=llm, embeddings=embeddings,
            run_config=RunConfig(timeout=180, max_retries=10, max_wait=60, max_workers=2),
        )
        return result.to_pandas().to_dict(orient="records")

    # Cache từng metric của từng câu ra đĩa (key = model + nội dung câu) để lỗi quota/timeout
    # giữa chừng không làm mất điểm đã chấm. Cache chỉ tái sử dụng kết quả RAGAS thật.
    # METRIC_VERSIONS đổi khi cách chấm một metric thay đổi → metric đó được chấm lại cho mọi câu.
    metric_versions = {"faithfulness": "all-sentences-v2"}
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              ".ragas_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    model_tag = "|".join(ragas_model_ids())
    keys = [hashlib.sha256(json.dumps([model_tag, questions[i], answers[i], list(contexts[i]),
                                       ground_truths[i]], ensure_ascii=False).encode("utf-8")).hexdigest()
            for i in range(len(questions))]
    rows: list[dict] = []
    for k in keys:
        cached = dict(cache.get(k, {}))
        for name, version in metric_versions.items():
            if cached.get(f"_{name}_version") != version:
                cached.pop(name, None)
        rows.append(cached)

    def save():
        for k, row in zip(keys, rows):
            entry = {n: row[n] for n in metric_names if _is_finite(row.get(n))}
            entry.update({f"_{n}_version": v for n, v in metric_versions.items() if n in entry})
            cache[k] = entry
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    batch_size = int(os.getenv("RAGAS_BATCH_SIZE", "10"))
    # Lần đầu + 2 lần chạy lại cho NaN tạm thời (timeout/parse lỗi); không gán điểm giả.
    for attempt in range(3):
        missing_by_metric = {n: [i for i, row in enumerate(rows) if not _is_finite(row.get(n))]
                             for n in metric_names}
        if not any(missing_by_metric.values()):
            break
        if attempt:
            print(f"  RAGAS NaN còn lại: { {n: len(v) for n, v in missing_by_metric.items() if v} }; chạy lại...")
        todo = sorted({i for v in missing_by_metric.values() for i in v})
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            names = [n for n in metric_names if any(i in missing_by_metric[n] for i in batch)]
            print(f"  RAGAS batch: {len(batch)} câu, metrics={names}", flush=True)
            for i, row in zip(batch, run(batch, names)):
                for n in names:
                    if not _is_finite(rows[i].get(n)):
                        rows[i][n] = row.get(n)
            save()
    still = [i for i, row in enumerate(rows) if not _all_finite(row, metric_names)]
    if still:
        raise RuntimeError(f"RAGAS vẫn trả NaN ở các index {still}; không thay bằng điểm giả.")

    per_question = [
        EvalResult(
            question=questions[index], answer=answers[index],
            contexts=list(contexts[index]), ground_truth=ground_truths[index],
            **{name: max(0.0, min(1.0, float(row[name]))) for name in metric_names},
        )
        for index, row in enumerate(rows)
    ]
    return _aggregate(per_question, backend=f"ragas_{provider_name()}",
                      evaluator_models=ragas_model_ids())


def _faithfulness_metric():
    """RAGAS 0.1.22 Faithfulness chỉ giữ câu kết thúc bằng '.', nên answer ngắn
    ("3 ngày làm việc", "Không") hoặc dạng list không có câu nào → luôn NaN.
    Subclass này giữ mọi câu không rỗng; prompt và cách chấm của RAGAS giữ nguyên."""
    from ragas.metrics._faithfulness import Faithfulness

    class _AllSentencesFaithfulness(Faithfulness):
        def _create_statements_prompt(self, row):
            assert self.sentence_segmenter is not None, "sentence_segmenter is not set"
            sentences = [s for s in self.sentence_segmenter.segment(row["answer"]) if s.strip()]
            return self.statement_prompt.format(
                question=row["question"], answer=row["answer"],
                sentences="\n".join(f"{i}:{x}" for i, x in enumerate(sentences)))

    return _AllSentencesFaithfulness()


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _all_finite(row: dict, names: list[str]) -> bool:
    return all(_is_finite(row.get(name)) for name in names)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời chứa thông tin chưa được context hỗ trợ.",
            "Siết prompt chỉ dùng context, giảm temperature và kiểm tra citation.",
        ),
        "context_recall": (
            "Retriever bỏ sót đoạn chứa đáp án.",
            "Điều chỉnh chunking, tăng candidate pool hoặc bổ sung BM25/query expansion.",
        ),
        "context_precision": (
            "Top-k chứa quá nhiều đoạn không liên quan.",
            "Tăng chất lượng reranking và thêm metadata/version filter.",
        ),
        "answer_relevancy": (
            "Câu trả lời chưa tập trung đúng ý câu hỏi.",
            "Dùng prompt trả lời trực tiếp và hậu xử lý câu trả lời ngắn gọn.",
        ),
    }
    analyzed = []
    for result in eval_results:
        metric_scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metric_scores, key=metric_scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "average_score": round(mean(metric_scores.values()), 4),
            "worst_metric": worst_metric,
            "score": round(metric_scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": _error_tree(result),
        })
    return sorted(analyzed, key=lambda item: item["average_score"])[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {
            key: float(results.get(key, 0.0))
            for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        },
        "num_questions": len(results.get("per_question", [])),
        "evaluation_backend": results.get("evaluation_backend", "unknown"),
        "latency_breakdown_ms": results.get("latency_breakdown_ms", {}),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


def _safe_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(score) else max(0.0, min(1.0, score))


def _aggregate(results: list[EvalResult], backend: str, evaluator_models: list[str]) -> dict:
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    aggregate = {
        metric: round(mean(getattr(result, metric) for result in results), 4) if results else 0.0
        for metric in metric_names
    }
    return {**aggregate, "evaluation_backend": backend, "per_question": results,
            "metadata": {"evaluator_models": list(evaluator_models), "backend": backend}}


def _offline_evaluate(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvalResult:
    answer_tokens = _content_tokens(answer)
    question_tokens = _content_tokens(question)
    truth_tokens = _content_tokens(ground_truth)
    context_token_sets = [_content_tokens(context) for context in contexts]
    all_context_tokens = set().union(*context_token_sets) if context_token_sets else set()

    faithfulness = _coverage(answer_tokens, all_context_tokens)
    answer_relevancy = max(
        _coverage(question_tokens, answer_tokens),
        _coverage(truth_tokens, answer_tokens),
    )
    relevance = [
        min(1.0, 1.35 * max(
            _coverage(question_tokens, tokens),
            _coverage(truth_tokens, tokens),
        ))
        for tokens in context_token_sets
    ]
    context_precision = _average_precision(relevance)
    context_recall = _coverage(truth_tokens, all_context_tokens)

    return EvalResult(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        faithfulness=round(faithfulness, 4),
        answer_relevancy=round(answer_relevancy, 4),
        context_precision=round(context_precision, 4),
        context_recall=round(context_recall, 4),
    )


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "ai", "bao", "bi", "cac", "cho", "co", "cua", "duoc", "gi", "la", "mot",
        "nao", "nhan", "nhieu", "o", "phai", "the", "thi", "trong", "va", "voi",
        "là", "có", "của", "được", "và", "cho", "trong", "không", "nhân", "viên",
    }
    return {
        token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stopwords
    }


def _coverage(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 1.0
    return len(expected & observed) / len(expected)


def _f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _average_precision(relevance: list[float]) -> float:
    if not relevance:
        return 0.0
    weighted = sum(score / (index + 1) for index, score in enumerate(relevance))
    normalizer = sum(1.0 / (index + 1) for index in range(len(relevance)))
    return min(1.0, weighted / normalizer)


def _error_tree(result: EvalResult) -> str:
    if result.context_recall < 0.5:
        return "Output sai -> Context thiếu -> Retrieval/chunking"
    if result.context_precision < 0.5:
        return "Output sai -> Context nhiễu -> Reranking/filter"
    if result.faithfulness < 0.5:
        return "Output sai -> Context đủ -> Generation hallucination"
    return "Output chưa sát -> Context đúng -> Prompt/answer formulation"


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
