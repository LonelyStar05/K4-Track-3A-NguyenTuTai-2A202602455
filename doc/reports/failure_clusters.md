# Phân tích lỗi Phase A

Nhóm: NguyenTuTai (cá nhân)

Người phụ trách: Nguyễn Tú Tài — 2A202602455

Nguồn: `artifacts/ragas_50q.json`; run timestamp: 2026-10-12T14:44:26.208441+00:00.
Generation mode/fallback và điều kiện chạy: cả 50 answers có `generation_mode=llm` (không có extractive fallback). Pipeline Day 18: hierarchical chunking (109 child chunks) → enrichment bằng LLM → hybrid search BM25 + dense (hashing encoder 1024 chiều, Qdrant in-memory vì Docker không chạy) → reranker lexical top-3 → generation với prompt "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'". Không dùng BGE-M3/bge-reranker thật. RAGAS 0.1.22 chạy theo batch 10 câu, có cache từng metric để chịu lỗi quota free tier; điểm đều là kết quả RAGAS thật, không có điểm điền tay.
Models generation/evaluator: generation = `gemini-3.5-flash-lite`; evaluator (từ `m4.evaluate_ragas()["metadata"]["evaluator_models"]`) = `gemini-flash-lite-latest` (LLM) + `gemini-embedding-001` (embedding).

Thay đổi đo lường cần biết: RAGAS 0.1.22 Faithfulness chỉ giữ các câu kết thúc bằng dấu "." nên các answer ngắn như "3 ngày làm việc", "Không" luôn ra NaN. Adapter m4 dùng một subclass giữ mọi câu không rỗng (prompt và cách tính điểm giữ nguyên), và faithfulness được chấm lại bằng cách này cho cả 50 câu để đồng nhất.

## Điểm theo distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
|---|---|---|---|
| faithfulness | 1.0000 | 0.7164 | 0.6750 |
| answer_relevancy | 0.8982 | 0.6342 | 0.7152 |
| context_precision | 0.8750 | 0.4750 | 0.7500 |
| context_recall | 0.7500 | 0.5250 | 0.6333 |
| avg_score | 0.8808 | 0.5877 | 0.6934 |

## Bottom 10 và nguyên nhân

Diagnosis/suggested_fix dưới đây là output nguyên văn của Task 3 (`bottom_10`), exporter giữ nguyên.

| rank | id | distribution | avg_score | worst_metric | diagnosis | suggested_fix |
|---|---|---|---|---|---|---|
| 1 | 40 | multi_hop | 0.0000 | faithfulness | Pipeline từ chối ('Không tìm thấy') vì retrieval thiếu chunk chứa đáp án (context_recall=0.00) — lỗi retrieval, không phải hallucination | Tăng top-k/trả parent chunk, tách câu hỏi nhiều bước thành sub-query trước khi search |
| 2 | 49 | adversarial | 0.0000 | faithfulness | Pipeline từ chối ('Không tìm thấy') vì retrieval thiếu chunk chứa đáp án (context_recall=0.00) — lỗi retrieval, không phải hallucination | Tăng top-k/trả parent chunk, tách câu hỏi nhiều bước thành sub-query trước khi search |
| 3 | 35 | multi_hop | 0.1668 | faithfulness | Pipeline từ chối ('Không tìm thấy') vì retrieval thiếu chunk chứa đáp án (context_recall=0.00) — lỗi retrieval, không phải hallucination | Tăng top-k/trả parent chunk, tách câu hỏi nhiều bước thành sub-query trước khi search |
| 4 | 6 | factual | 0.2500 | answer_relevancy | Pipeline từ chối ('Không tìm thấy') vì retrieval thiếu chunk chứa đáp án (context_recall=0.00) — lỗi retrieval, không phải hallucination | Tăng top-k/trả parent chunk, tách câu hỏi nhiều bước thành sub-query trước khi search |
| 5 | 25 | multi_hop | 0.2500 | answer_relevancy | Pipeline từ chối ('Không tìm thấy') vì retrieval thiếu chunk chứa đáp án (context_recall=0.00) — lỗi retrieval, không phải hallucination | Tăng top-k/trả parent chunk, tách câu hỏi nhiều bước thành sub-query trước khi search |
| 6 | 27 | multi_hop | 0.2500 | answer_relevancy | Pipeline từ chối ('Không tìm thấy') vì retrieval thiếu chunk chứa đáp án (context_recall=0.00) — lỗi retrieval, không phải hallucination | Tăng top-k/trả parent chunk, tách câu hỏi nhiều bước thành sub-query trước khi search |
| 7 | 33 | multi_hop | 0.2500 | answer_relevancy | Answer doesn't match question — câu cần ghép nhiều đoạn, top-k hiện tại chưa phủ đủ | Improve prompt template; tăng top-k hoặc lấy parent chunk cho câu nhiều bước |
| 8 | 21 | multi_hop | 0.3750 | faithfulness | Context đã chứa đáp án (context_recall=0.50) nhưng LLM vẫn từ chối — prompt quá bảo thủ, không tổng hợp/tính toán từ nhiều đoạn | Sửa prompt cho phép suy luận và tính toán từ context, thêm few-shot câu multi-hop |
| 9 | 42 | adversarial | 0.3750 | faithfulness | Context đã chứa đáp án (context_recall=1.00) nhưng LLM vẫn từ chối — prompt quá bảo thủ, không tổng hợp/tính toán từ nhiều đoạn | Sửa prompt cho phép suy luận và tính toán từ context, thêm few-shot câu multi-hop |
| 10 | 50 | adversarial | 0.5472 | faithfulness | LLM hallucinating — câu bẫy phiên bản/phủ định, dễ lấy nhầm chính sách cũ | Tighten system prompt, lower temperature; ưu tiên tài liệu mới nhất (metadata version/effective date) khi rerank |

Đối chiếu question/answer/contexts/reference:

- **Lỗi retrieval là nhóm lớn nhất (id 40, 49, 35, 6, 25, 27).** Cả 6 câu trả "Không tìm thấy." với context_recall = 0. Ví dụ id 35 (Junior 12 triệu đang thử việc) cần ghép 85% lương thử việc từ `thu_viec.md` với phụ cấp ăn trưa từ `phu_cap.md`; top-3 chỉ có bảng lương, lương tháng 13 và quyền lợi thử việc, thiếu `phu_cap.md`. Id 40 hỏi về vi phạm bảo mật của Manager đang thử việc, nhưng 3 context là thời gian lương thử việc, phạm vi chính sách làm việc từ xa và khung lương thử việc, không có `bao_mat_su_co.md`: các từ "Manager/thử việc/90 ngày" kéo BM25 và hashing encoder về sai tài liệu. Đây là lỗi retrieval chứ không phải hallucination; generation từ chối là hành vi đúng với prompt.
- **Context có đáp án nhưng LLM vẫn từ chối (id 21, 42).** Id 42: context thứ hai là phần thâm niên của `nghi_phep_nam_v2024.md` (context_recall = 1.0), nhưng model vẫn trả "Không tìm thấy.". Id 21 cần tính 15 + 9÷3 = 18 ngày, rồi ghép khung lương Senior. Prompt "CHỈ dựa trên context" khiến model lite không dám suy ra hoặc tính toán.
- **Answer lệch câu hỏi (id 33).** Câu hỏi về tổng phụ cấp và phép năm của Manager 12 năm thâm niên; answer chỉ liệt kê 15 ngày phép v2024 và so sánh với v2023, không tính 4 ngày thâm niên và không có phụ cấp, vì cả 3 context đều từ tài liệu nghỉ phép.
- **Faithfulness = 0 với answer đúng (id 50).** Answer "Không." khớp reference, và context đầu tiên chứa đúng câu "Không sử dụng VPN cá nhân (NordVPN, ExpressVPN) trong mạng công ty". Retrieval và generation đều đúng; faithfulness = 0 là lỗi của evaluator với answer một từ (không tách được statement nào có thể kiểm chứng). Nhãn "LLM hallucinating" mà DIAGNOSTIC_TREE gán cho câu này là sai, và cho thấy giới hạn của việc chẩn đoán chỉ dựa trên worst_metric.
- **Faithfulness của các câu từ chối là nhiễu evaluator.** Trên 10 answer "Không tìm thấy.", faithfulness ra 1.0 ở 5 câu và 0.0 ở 5 câu, dù nội dung giống hệt nhau. Vì vậy Task 3 chẩn đoán các câu này theo context_recall thay vì theo nhãn "LLM hallucinating".

## Matrix và nhóm yếu nhất

Mỗi câu được đếm một lần theo worst_metric của nó; tổng mỗi cột chỉ là kích thước nhóm (20/20/10), không phải số failure.

| worst_metric | factual | multi_hop | adversarial | tổng |
|---|---|---|---|---|
| faithfulness | 0 | 8 | 3 | 11 |
| answer_relevancy | 11 | 4 | 1 | 16 |
| context_precision | 2 | 5 | 1 | 8 |
| context_recall | 7 | 3 | 5 | 15 |

Nhóm yếu theo mean thấp nhất: multi_hop; mean: 0.5877 (factual 0.8808, adversarial 0.6934).
Worst metric xuất hiện nhiều nhất: answer_relevancy; số câu: 16.

Multi_hop yếu vì mỗi câu cần 2–3 tài liệu (lương + thử việc + phụ cấp, phép năm + nghỉ ốm), trong khi top-3 sau rerank thường chỉ phủ một tài liệu: context_precision 0.475 và context_recall 0.525 thấp nhất trong ba nhóm, và 7/10 câu "Không tìm thấy." thuộc multi_hop. Answer_relevancy đứng đầu matrix chủ yếu do 11 câu factual có mọi metric khác cao (faithfulness factual = 1.0), nên metric thấp nhất của chúng chỉ là answer_relevancy ở mức 0.8–0.9. Đây là chênh lệch nhỏ, không phải lỗi; không nên đọc matrix như danh sách lỗi.

Đề xuất sửa và cách kiểm chứng:

1. Retrieval cho multi-hop: tách câu hỏi thành sub-query (một query cho mỗi thực thể/chính sách), hợp kết quả, tăng RERANK_TOP_K từ 3 lên 5 hoặc trả parent chunk (2048 ký tự) thay cho child chunk. Bật BGE-M3 + bge-reranker thật (`RAG_USE_TRANSFORMERS=1`) thay cho hashing/lexical.
2. Prompt generation: cho phép tính toán và tổng hợp từ nhiều đoạn, chỉ trả "Không tìm thấy" khi không đoạn nào liên quan; thêm 1–2 ví dụ few-shot multi-hop.
3. Kiểm chứng: chạy lại `python scripts/setup_answers.py --live` rồi `run_experiments.py --phase a --live` với cùng evaluator; so sánh multi_hop context_recall (0.525), số câu "Không tìm thấy." (10/50) và avg_score multi_hop (0.588) trước/sau. Chỉ coi là cải thiện khi factual không giảm.

## Bẫy adversarial và giới hạn

Adversarial mean (0.693) thấp hơn factual (0.881) nhưng cao hơn multi_hop (0.588). Trong bottom 10 có 3 câu adversarial:

- id 49 (hỏi cả v2023 và bản đang hiệu lực): top-3 chỉ có tài liệu v2023 (12 ngày), nghỉ không lương và thủ tục đăng ký; không có `nghi_phep_nam_v2024.md`. Câu hỏi chứa "cũ (v2023)" nên `_version_bonus` của reranker Day 18 còn cộng điểm cho bản cũ, và bản v2024 bị đẩy ra ngoài. Model trả "Không tìm thấy." thay vì nói sai bản hiện hành: an toàn, nhưng đây đúng là lỗi version-conflict ở tầng retrieval. Sửa: với câu hỏi so sánh phiên bản, truy xuất riêng từng phiên bản (một sub-query mỗi version) thay vì cộng điểm một chiều.
- id 42 (thâm niên bao nhiêu năm thì cộng phép): context có đoạn v2024 nhưng model vẫn từ chối, nên đây là lỗi prompt, không phải nhầm v2023 với v2024.
- id 50 (VPN cá nhân, bẫy mâu thuẫn chính sách): pipeline trả đúng "Không." dựa trên context có câu cấm VPN cá nhân; câu này nằm trong bottom 10 chỉ vì faithfulness của evaluator ra 0 với answer một từ.

Ở chiều ngược lại, câu pairwise id 41 (Phase B) cho thấy pipeline nêu đúng v2024 = 15 ngày và nhắc v2023 = 12 ngày. Reranker Day 18 có `_version_bonus` ưu tiên tài liệu v2024 và phạt tài liệu đã thay thế, nên với câu hỏi thông thường (id 41) pipeline chọn đúng bản hiện hành; nhưng khi câu hỏi nhắc chính sách cũ (id 49), cùng heuristic đó lại đẩy bản v2024 ra khỏi top-3. Không có answer nào trong bottom 10 khẳng định sai phiên bản; lỗi adversarial là thiếu context, từ chối hoặc nhiễu evaluator. Điểm adversarial thấp hơn factual chỉ cho thấy nhóm này khó hơn; tự nó không chứng minh pipeline đã nhận ra version conflict.

Giới hạn: chỉ 50 câu (adversarial có 10), một lượt chạy, evaluator là model lite free tier (`gemini-flash-lite-latest`) với kết quả không ổn định trên answer từ chối; embedding/reranker là bản fallback nhẹ, không phải BGE-M3. Các mean chênh nhau dưới khoảng 0.05 không nên coi là khác biệt thật.
