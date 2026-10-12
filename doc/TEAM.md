# Nhóm thực hiện K4 Track 3A

Tên nhóm: NguyenTuTai (làm cá nhân, một thành viên)

Repo chung: https://github.com/LonelyStar05/K4-Track-3A-NguyenTuTai-2A202602455

Trưởng nhóm/người nộp: Nguyễn Tú Tài — 2A202602455

Ngày lab/deadline áp dụng: 23:59 ngày diễn ra lab theo thông báo lớp — Asia/Ho_Chi_Minh (UTC+07:00)

Commit dùng để chấm: commit mới nhất trên nhánh main tại thời điểm nộp link (ghi SHA khi gửi bài qua vlearn).

Mỗi nhóm nộp một repo. Bài này do một người thực hiện nên bảng chỉ có một dòng.

## Thành viên và phân công

| STT | Họ và tên | MSSV | Vai trò / công việc thật | Bằng chứng |
|---|---|---|---|---|
| 1 | Nguyễn Tú Tài | 2A202602455 | Toàn bộ: tích hợp m1–m5 Day 18, Tasks 1–13 trong `src/phase_a_ragas.py`, `src/phase_b_judge.py`, `src/phase_c_guard.py`, chạy batch A/B/C, viết báo cáo | Lịch sử commit, `artifacts/*.json`, `doc/reports/*.md` |

Người tích hợp/chạy batch/nộp link: Nguyễn Tú Tài — chép m1–m5 từ repo Day 18, viết adapter (m4 metadata, Gemini provider, qdrant-client 1.9), chạy `setup_answers.py --live` và ba runner `--live`, chạy `check_lab.py`, push và nộp link.

## Môi trường và chuẩn bị trước lớp

OS/Python/compiler: Windows 11 Home 10.0.26200; Python 3.11.9 (`py -3.11 -m venv .venv`); không có MSVC Build Tools.
Virtualenv/dependencies/imports: `pip install -r requirements/requirements.txt -c requirements/constraints-core.txt` lỗi khi build `annoy==1.17.3` (thiếu Microsoft Visual C++ 14.0). Đã cài mọi dependency còn lại, `nemoguardrails==0.10.0 --no-deps` cùng các dependency của nó trừ annoy, pin `pyarrow<20` để tương thích NumPy 1.26.4. `import ragas, nemoguardrails, presidio_analyzer, spacy` chạy OK (ragas 0.1.22, NeMo 0.10.0). `pip check` vẫn báo thiếu annoy; NeMo log traceback khi khởi tạo bot-message embedding index nhưng self-check input/output rails (không cần annoy) chạy bình thường. `python -m spacy download en_core_web_lg` thành công.
Models/Qdrant/50 answers đã chuẩn bị: LLM là Gemini qua OpenAI-compatible endpoint (free tier, 15 RPM/model) — generation `gemini-3.5-flash-lite`, enrichment `gemini-3.1-flash-lite`, RAGAS evaluator `gemini-flash-lite-latest` + `gemini-embedding-001` (lượt RAGAS đầu dùng `gemini-3.1-flash-lite` nhưng hết quota 500 request/ngày; chạy lại toàn bộ với key Gemini thứ hai, có cache từng metric), judge `gemini-3.1-flash-lite-preview` (lần đầu dùng `gemini-3.5-flash` nhưng hết quota 20 request/ngày), guard `gemini-3.5-flash-lite`. Docker Desktop không chạy nên m2 dùng Qdrant in-memory (`QdrantClient(":memory:")`); embedding dense dùng hashing encoder 1024 chiều và reranker lexical của Day 18 thay cho BGE-M3/bge-reranker (không tải model ~4.5GB). 50 answers đều `generation_mode=llm` (lần chạy đầu có 18 fallback do 429 nên đã thêm retry và chạy lại).

## INDIVIDUAL

### NguyenTuTai-2A202602455

- Phần được giao và phần thực tế đã làm: toàn bộ lab. Phase A: `group_by_distribution`, `run_ragas_50q` (kiểm tra độ dài, NaN, metadata evaluator_models), `bottom_10` (diagnosis theo worst_metric, distribution và phân biệt câu từ chối do retrieval thiếu với do prompt quá bảo thủ), `cluster_analysis` (matrix, nhóm yếu theo mean). Phase B: `pairwise_judge` (prompt có reference, JSON validated), `swap_and_average`, `cohen_kappa`, `bias_report`. Phase C: `pii_scan` (chỉ giữ entity có pattern rõ để tránh false positive NER), `check_input_rail`, `check_output_rail`, `run_adversarial_suite`, `measure_p95_latency`. Tích hợp m1–m5, sửa m4 để báo model, không lặng lẽ dùng proxy offline, cache RAGAS theo metric và sửa lỗi Faithfulness của RAGAS 0.1.22 (bỏ mọi câu không kết thúc bằng '.' → NaN với answer ngắn), đăng ký provider `openai_chat` cho NeMo.
- Bằng chứng: các commit trong repo; `artifacts/answers_50q.json`, `ragas_50q.json`, `judge_results.json`, `guard_results.json`; ba báo cáo trong `doc/reports/`.
- Cách kiểm tra: `python -m pytest tests/ -q` (86 passed, mock, không mạng); ba runner `scripts/run_experiments.py --phase a|b|c --live`; `python scripts/check_lab.py`.
- Điều đã học: RAGAS tách lỗi retrieval (context precision/recall) với lỗi generation (faithfulness/relevancy); matrix worst_metric chỉ đếm câu, nhóm yếu phải chọn theo mean; judge pairwise cần swap để lộ position bias, còn nhãn 0/1 cần κ để so với người; guardrail phải có clean controls vì always-block cũng đạt 20/20 attacks; latency phải nói rõ phạm vi đo.
- Giới hạn/cách xử lý: free tier Gemini 15 RPM/model → tách model theo vai trò, tăng `max_retries`, nghỉ 4.5s giữa các mẫu latency (ngoài vùng đo); Gemini 3.x trả rỗng với `max_tokens=3` của NeMo self-check → đặt `max_tokens: 50` trong prompt config; NeMo 0.10 chỉ dùng chat API cho tên model gpt-* → đăng ký provider `openai_chat` trong `guardrails/config.py`. Không có annoy/MSVC, không có Docker Qdrant, không dùng BGE-M3 thật.
- Nguồn/AI hỗ trợ nếu có: Claude Code hỗ trợ viết code, adapter và nháp báo cáo; mọi số liệu trong báo cáo lấy trực tiếp từ JSON do runner tạo, đã đối chiếu bằng `check_lab.py`. Module m1–m5 lấy từ bài Day 18 của chính tôi.

## Cập nhật sau khi nộp

Chưa có cập nhật sau khi nộp; nếu sửa sau deadline sẽ ghi commit, thời điểm và lý do tại đây.
