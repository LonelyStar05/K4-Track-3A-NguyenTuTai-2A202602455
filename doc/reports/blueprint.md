# Blueprint CI/CD cho RAG và guardrails

Nhóm: NguyenTuTai (cá nhân)

Người phụ trách: Nguyễn Tú Tài — 2A202602455

Đây là **đề xuất CI/CD**, không phải workflow đã được triển khai/chạy.
Nguồn kết quả: artifacts/ragas_50q.json, judge_results.json, guard_results.json.
Run timestamp: 2026-10-12T12:58:27.697517+00:00 (guard_results.json); guard models: `gemini-3.5-flash-lite`.
Models generation/evaluator/judge từ A/B: generation `gemini-3.5-flash-lite` (answers_50q.json, ghi trong cả ragas_50q.json và judge_results.json); evaluator `gemini-flash-lite-latest` + `gemini-embedding-001` (ragas_50q.json, run 2026-10-12T14:44:26Z); judge `gemini-3.1-flash-lite-preview` (judge_results.json, run 2026-10-12T13:02:39Z). Tất cả gọi qua Gemini OpenAI-compatible endpoint, free tier.

## Kiến trúc và failure action

Input → Presidio → NeMo input → RAG → NeMo output → response.
Request thực dừng sớm khi bị chặn; khác phép đo input layers chạy đủ hai tầng.

Data flow và failure action đề xuất:

1. **Presidio (local, khoảng 7–10 ms):** recognizer VN_CCCD (12/9 số), VN_PHONE (0[3-9] + 8 số) và EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IBAN, SSN, passport, IP. Chỉ chặn theo các entity có pattern rõ; bỏ PERSON/LOCATION/DATE_TIME của spaCy tiếng Anh vì chúng gán nhãn nhầm trên câu tiếng Việt và gây chặn nhầm. Có PII → chặn request, trả câu từ chối chuẩn, log bản `anonymized` (ví dụ `<VN_CCCD>`), không log input gốc.
2. **NeMo input rail (self check input):** chặn off-topic, jailbreak, prompt injection, giả danh thẩm quyền, trích system prompt, yêu cầu PII/lương cá nhân. Quyết định đọc từ `result.log.activated_rails[*].stop`, không dò từ khóa "xin lỗi" trong câu trả lời. Bị chặn → trả câu `bot refuse to respond` trong `rails.co`, log id request + tên rail.
3. **RAG (m1–m5):** hybrid search → rerank top-3 → generation chỉ dựa trên context. Không có context → trả "Không tìm thấy." (không gọi LLM). LLM lỗi → extractive fallback, đánh dấu `generation_mode=extractive_fallback`.
4. **NeMo output rail (self check output):** answer chứa CCCD, số điện thoại/email cá nhân, lương cá nhân hoặc credential → thay bằng câu từ chối (`final_answer != answer`); answer sạch giữ nguyên.
5. **Fail-closed:** nếu NeMo/LLM guard lỗi hoặc timeout, request bị chặn với thông báo thử lại, kèm cảnh báo cho người trực; không trả answer chưa qua output rail.

Secrets: key API chỉ nằm trong `.env`/biến môi trường (đã gitignore); `.env.example` chỉ có tên biến. Trong CI dùng secret store của runner, không in key ra log. Guard dùng model riêng (`gemini-3.5-flash-lite`) để quota của guard không tranh với generation/evaluator.

## Phạm vi và số đo latency

Bắt buộc Task 12: **input_layers_sequential**, Presidio + NeMo input, không
short-circuit. Engines tạo trước đo; có hai warm-up thật không tính vào mẫu.
Percentile nội suy tuyến tính từ raw samples; tổng từ tổng từng sample.

Số mẫu timed: 20; warm-up: 2;
phương pháp: linear.

| Layer / scope | P50 ms | P95 ms | P99 ms | Tình trạng |
|---|---|---|---|---|
| Presidio | 6.99 | 9.69 | 15.19 | Đo Task 12, rất nhỏ so với budget |
| NeMo input | 827.64 | 1332.84 | 1343.91 | Đo Task 12, chiếm gần toàn bộ thời gian |
| Input guard sequential | 834.19 | 1348.94 | 1352.90 | Đo Task 12, vượt target 500 ms |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| Full guard / short-circuit request | chưa đo | chưa đo | chưa đo | Không suy từ input_guard_ms |

Input-layer budget target: 500ms; kết quả thực tế/budget_ok: input_guard P95 = 1348.94 ms, `latency_budget_ok = false`.

Bottleneck là NeMo input rail: mỗi lượt là một lời gọi LLM qua mạng tới Gemini free tier (khoảng 650–1350 ms/lần), còn Presidio chỉ khoảng 10 ms. Giữa các mẫu có nghỉ 4.5 s *ngoài vùng đo* để không chạm giới hạn 15 request/phút; nếu không, thời gian SDK retry 429 sẽ lẫn vào số đo. Số đo phản ánh độ trễ mạng và hàng đợi free tier, không phải chi phí tính toán của rail. Hướng giảm: dùng endpoint trả phí/gần hơn; chặn sớm các pattern jailbreak rõ ràng bằng classifier local trước khi gọi LLM; cache quyết định cho câu hỏi lặp lại; chạy Presidio và NeMo song song vì Presidio chỉ chặn chứ không biến đổi input.

Kế hoạch đo các phần "chưa đo": thêm timestamps `perf_counter` quanh search, rerank, generation và output rail trong cùng một request; 2 warm-up + ít nhất 50 request thật cycling bộ 50 câu; báo P50/P95/P99 cho từng tầng và cho toàn request (gồm trường hợp short-circuit khi input bị chặn); không cộng P95 từng tầng để suy ra P95 tổng.

## CI gates đề xuất

- RAGAS faithfulness ≥0.75, avg_score ≥0.65 trên 50 câu.
- Attacks ≥18/20 cho CI đề xuất; clean 8/8 allowed; output cases đúng.
- Full guard P95 <500ms chỉ áp dụng sau khi thực sự đo full guard.

Mốc bắt buộc lab vẫn attacks ≥15/20; bonus ≥18/20 kèm clean controls đạt.

Job đề xuất (chưa triển khai; repo này không có workflow CI nào đang chạy):

1. `unit`: `python -m pytest tests/ -q` (offline, mock) trên mọi PR.
2. `eval-nightly` (cần secret API, chạy theo lịch hoặc khi đổi `src/m*`, prompt, `guardrails/`): `python scripts/setup_answers.py --live`, rồi `python scripts/run_experiments.py --phase a --live`, `--phase b --live`, `--phase c --live --latency-samples 20`.
3. `gate`: `python scripts/check_lab.py`, cộng thêm một script gate mới (ví dụ `scripts/ci_gate.py`) đọc ba JSON và `sys.exit(1)` khi: overall faithfulness < MIN_FAITHFULNESS (0.75), avg_score toàn bộ < MIN_AVG (0.65), attacks < 18/20, clean false_positive_rate > 0, có output case sai, hoặc κ < 0.6. Các biến MIN_* chỉ có tác dụng khi script này thực sự được viết và chạy; hiện chưa có.

Đối chiếu kết quả hiện tại với gate đề xuất: faithfulness toàn bộ ≈ 0.822 (đạt ≥0.75); avg_score toàn bộ ≈ 0.726 (đạt ≥0.65), nhưng multi_hop avg 0.588 thấp hơn 0.65, nên nên thêm gate theo từng distribution; attacks 20/20, clean 8/8, output 4/4 (đạt); κ = 1.0 (đạt); input guard P95 1349 ms (không đạt target 500 ms, chưa áp gate full guard vì chưa đo).

## Monitoring và cải tiến

Monitoring đề xuất cho production:

- **Guard:** tỷ lệ block theo rail và category mỗi giờ; cảnh báo khi block rate lệch quá 2 lần so với baseline 7 ngày (có thể là tấn công hàng loạt, hoặc prompt/model drift gây chặn nhầm). Lấy mẫu 1% request bị chặn (bản anonymized) cho người xem lại; mọi khiếu nại "bị chặn nhầm" được thêm vào clean set.
- **Chất lượng RAG:** hằng tuần lấy mẫu 50 request thật (đã ẩn PII), chạy RAGAS với cùng evaluator; cảnh báo khi faithfulness < 0.75 hoặc tỷ lệ answer "Không tìm thấy." > 15% (hiện là 10/50 = 20%).
- **Judge drift:** mỗi tháng gán nhãn người cho 20 câu mới, đo lại κ; κ < 0.6 thì không dùng judge làm gate cho đến khi sửa prompt/model.
- **Latency:** P50/P95/P99 từng tầng và toàn request từ timestamps; cảnh báo P95 input guard > 1500 ms hoặc lỗi 429 > 1%.
- **Privacy:** log chỉ chứa bản anonymized; chính sách lưu log 30 ngày; người trực guardrail chịu trách nhiệm xử lý cảnh báo.

RAGAS mean/worst distribution: factual 0.881, multi_hop 0.588 (yếu nhất), adversarial 0.693.
Cohen’s κ: 1.0 (10/10 nhãn khớp human, judge `gemini-3.1-flash-lite-preview`).
Attacks passed: 20/20; clean false_positive_rate: 0.0.

Giới hạn bằng chứng: bộ 50/20/8/4 nhỏ và cố định; attacks 20/20 trên 20 mẫu đã biết không đảm bảo chặn được biến thể mới; κ = 1.0 chỉ trên 10 câu dễ phân định.

Cải tiến có thể kiểm tra, gắn với bằng chứng:

1. Tách câu hỏi multi-hop thành sub-query và tăng top-k 3→5 hoặc trả parent chunk; kỳ vọng multi_hop context_recall tăng từ 0.525 và số "Không tìm thấy." giảm từ 10/50 (đo bằng chạy lại setup + Phase A).
2. Sửa prompt generation cho phép tính toán/tổng hợp (id 21, 42 có context nhưng vẫn từ chối); đo số câu từ chối có context_recall ≥ 0.5.
3. Truy xuất riêng từng phiên bản cho câu hỏi so sánh version (id 49 bị `_version_bonus` đẩy mất v2024); thêm test hồi quy cho cặp v2023/v2024.
4. Thêm 20 biến thể attack mới (paraphrase, mã hóa, đa ngôn ngữ) và 20 câu sạch khó (có chữ "mật khẩu", "lương", "bỏ qua"); giữ gate attacks ≥90% và false positive = 0.
5. Đo full request latency (RAG + output rail) theo kế hoạch ở trên trước khi áp gate P95 < 500 ms; thử chạy Presidio song song với NeMo input và đo lại.
