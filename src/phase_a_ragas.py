from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH, ARTIFACTS_DIR
from src.experiment_support import write_json

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

@dataclass
class RagasEvaluation:
    """Task 2 output: measured rows plus model identifiers reported by m4."""
    results: list[RagasResult]
    evaluator_models: list[str]

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python scripts/setup_answers.py --live"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict]) -> dict:
    """Provided exporter: preserve the student-produced Task 3 rows verbatim."""
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_question": [asdict(r) for r in results],
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom,
    }
    return report


def save_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict],
                        path: str = os.path.join(ARTIFACTS_DIR, "ragas_50q.json")) -> None:
    write_json(path, build_phase_a_report(results, clusters, bottom))



# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        groups[item["distribution"]].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> RagasEvaluation:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gợi ý — import từ Day 18 của bạn:
        from src.m4_eval import evaluate_ragas

    Steps:
        1. Extract questions, answers, contexts, ground_truths từ answers list
        2. Gọi evaluate_ragas() từ m4_eval.py
        3. Kết hợp kết quả với distribution info từ answers list
        4. Return RagasEvaluation(results, raw["metadata"]["evaluator_models"])
           m4 must report its actual LLM/embedding models; never guess JUDGE_MODEL.
    """
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("⚠️  Không tìm thấy src/m4_eval.py — đã copy từ Day 18 chưa?")
        raise ValueError("Missing m4 evaluator; cannot declare its actual models")

    questions     = [a["question"]    for a in answers]
    ans_texts     = [a["answer"]      for a in answers]
    contexts      = [a["contexts"]    for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]

    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_q = raw.get("per_question", [])
    if len(per_q) != len(answers):
        raise ValueError(f"m4 trả {len(per_q)} kết quả cho {len(answers)} câu; không zip âm thầm.")
    evaluator_models = (raw.get("metadata") or {}).get("evaluator_models")
    if not evaluator_models:
        raise ValueError("m4 không báo metadata.evaluator_models; sửa adapter m4, không đoán model.")

    def metric(record, name: str) -> float:
        value = record[name] if isinstance(record, dict) else getattr(record, name)
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}={value} không hợp lệ (NaN/ngoài [0,1]).")
        return value

    results = []
    for a, pq in zip(answers, per_q):
        results.append(RagasResult(
            question_id=a["id"], distribution=a["distribution"],
            question=a["question"], answer=a["answer"],
            contexts=a["contexts"], ground_truth=a["ground_truth"],
            faithfulness=metric(pq, "faithfulness"),
            answer_relevancy=metric(pq, "answer_relevancy"),
            context_precision=metric(pq, "context_precision"),
            context_recall=metric(pq, "context_recall"),
        ))
    return RagasEvaluation(results, list(evaluator_models))


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    sorted_asc = sorted(results, key=lambda r: r.avg_score)
    bottom = sorted_asc[:10]
    output = []
    for i, r in enumerate(bottom):
        diag, fix = DIAGNOSTIC_TREE[r.worst_metric]
        if r.answer.strip().lower().startswith("không tìm thấy"):
            # Refusal: faithfulness của câu từ chối là nhiễu evaluator, không phải hallucination.
            if r.context_recall < 0.5:
                diag = (f"Pipeline từ chối ('Không tìm thấy') vì retrieval thiếu chunk chứa đáp án "
                        f"(context_recall={r.context_recall:.2f}) — lỗi retrieval, không phải hallucination")
                fix = "Tăng top-k/trả parent chunk, tách câu hỏi nhiều bước thành sub-query trước khi search"
            else:
                diag = (f"Context đã chứa đáp án (context_recall={r.context_recall:.2f}) nhưng LLM vẫn "
                        f"từ chối — prompt quá bảo thủ, không tổng hợp/tính toán từ nhiều đoạn")
                fix = "Sửa prompt cho phép suy luận và tính toán từ context, thêm few-shot câu multi-hop"
        elif r.distribution == "adversarial":
            diag += " — câu bẫy phiên bản/phủ định, dễ lấy nhầm chính sách cũ"
            fix += "; ưu tiên tài liệu mới nhất (metadata version/effective date) khi rerank"
        elif r.distribution == "multi_hop":
            diag += " — câu cần ghép nhiều đoạn, top-k hiện tại chưa phủ đủ"
            fix += "; tăng top-k hoặc lấy parent chunk cho câu nhiều bước"
        if not r.contexts:
            diag += " (không có context nào được truy xuất)"
        output.append({
            "rank": i + 1,
            "question_id": r.question_id,
            "distribution": r.distribution,
            "question": r.question,
            "avg_score": round(r.avg_score, 4),
            "worst_metric": r.worst_metric,
            "diagnosis": diag,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Matrix thống kê worst_metric của mọi câu, không coi mọi câu là một failure.
    Nhóm yếu nhất = distribution có avg_score trung bình thấp nhất (bỏ nhóm rỗng).
    Tổng matrix theo cột chỉ là kích thước nhóm 20/20/10, không đo chất lượng.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for r in results:
        matrix[r.worst_metric][r.distribution] += 1

    # Find dominant failure
    means = {
        d: sum(r.avg_score for r in results if r.distribution == d) /
           sum(r.distribution == d for r in results)
        for d in ("factual", "multi_hop", "adversarial")
        if any(r.distribution == d for r in results)
    }
    dominant_dist = min(means, key=means.get)  # lowest quality, NOT largest group
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    means_text = ", ".join(f"{d}={v:.3f}" for d, v in means.items())
    insight = (f"Distribution '{dominant_dist}' có điểm trung bình thấp nhất ({means_text}). "
               f"Metric '{dominant_metric}' là worst_metric của "
               f"{sum(matrix[dominant_metric].values())}/{len(results)} câu "
               f"({DIAGNOSTIC_TREE[dominant_metric][0]}). "
               f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}")

    return {"matrix": matrix, "distribution_avg_scores": means,
            "dominant_failure_distribution": dominant_dist,
            "dominant_failure_metric": dominant_metric, "insight": insight}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "a", *sys.argv[1:]]))
