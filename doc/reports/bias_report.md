# Đánh giá LLM Judge Phase B

Nhóm: NguyenTuTai (cá nhân)

Người phụ trách: Nguyễn Tú Tài — 2A202602455

Nguồn: `artifacts/judge_results.json`; run timestamp: 2026-10-12T13:02:39.170367+00:00.
Models generation/judge: generation = `gemini-3.5-flash-lite`; judge = `gemini-3.1-flash-lite-preview` (cùng model cho pairwise `phase_b_judge.JUDGE_MODEL` và binary helper `config.JUDGE_MODEL`), gọi qua Gemini OpenAI-compatible endpoint, temperature 0, JSON mode.

Ghi chú vận hành: lần chạy đầu dùng `gemini-3.5-flash` làm judge nhưng model này chỉ có 20 request/ngày ở free tier và hết quota giữa chừng, runner dừng trước khi ghi file. Kết quả dưới đây là của lần chạy lại duy nhất đã lưu, không chọn lọc lại sau khi xem kết quả.

## Năm cặp A/B và hai lượt

Nguồn cố định: A từ artifacts/answers_50q.json:answer; B từ
fixtures/human_labels_10q.json:model_answer; reference từ fixtures/test_set_50q.json:ground_truth.
Ids: 1, 5, 12, 21, 41. Không đổi nguồn sau khi xem kết quả judge.

| id | Candidate A (pipeline) | Candidate B (model_answer) | pass1 | pass2 (quy về A/B gốc) | final | consistent | scores pass1 | scores pass2 |
|---|---|---|---|---|---|---|---|---|
| 1 | "3 ngày làm việc" (15 ký tự) | "Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn." (57) | B | B | B | true | A 0.8 / B 1.0 | A 0.7 / B 1.0 |
| 5 | "Tổng Giám đốc (CEO)" (19) | "Cần Giám đốc phòng ban phê duyệt." (33) | A | A | A | true | A 1.0 / B 0.0 | A 1.0 / B 0.0 |
| 12 | "…là **1 tháng lương**." kèm điều kiện (89) | "Nhân viên được thưởng Tết tối thiểu 1 tháng lương." (50) | A | A | A | true | A 1.0 / B 0.7 | A 1.0 / B 0.7 |
| 21 | "Không tìm thấy." (15) | "Được nghỉ 18 ngày phép (15 cơ bản + 3 thâm niên). Lương Senior từ 20 đến 35 triệu." (82) | B | B | B | true | A 0.0 / B 1.0 | A 0.0 / B 1.0 |
| 41 | "…2024 là **15 ngày** (và là **12 ngày** theo chính sách năm 2023)." (135) | "Nhân viên được nghỉ 12 ngày phép năm." (37) | A | A | A | true | A 0.9 / B 0.2 | A 0.95 / B 0.2 |

Tóm tắt reasoning (đầy đủ trong JSON):

- id 1: cả hai đúng số ngày; B đủ hơn vì nêu "có lương". Cả hai lượt chọn B.
- id 5: A đúng theo reference (>50 triệu cần Tổng Giám đốc); B sai thẩm quyền.
- id 12: A nêu đủ điều kiện "chính thức, từ 6 tháng"; judge cho rằng B bỏ điều kiện nên điểm thấp hơn (0.7) dù con số đúng.
- id 21: A của pipeline trả "Không tìm thấy." dù reference có đáp án (lỗi retrieval câu multi-hop); B đúng cả 18 ngày phép và khung 20–35 triệu.
- id 41: A dùng đúng v2024 (15 ngày) và nhắc v2023; B dùng chính sách cũ 12 ngày → sai.

Position bias count/rate: 0/5 = 0.0.
Verbosity win rate: 0.8 (4/5); decisive cases: 5 (a_wins_a_longer = 2, b_wins_b_longer = 2).

Không có cặp nào đổi kết luận khi hoán đổi thứ tự, nên với mẫu này không thấy dấu hiệu position bias. Verbosity rate 0.8 vượt ngưỡng cảnh báo 0.6, nhưng đọc nội dung thì cả 4 trường hợp answer dài hơn thắng đều có lý do thực chất: id 1 và id 12 bản dài thêm điều kiện/quyền lợi có trong reference; id 21 bản ngắn là "Không tìm thấy."; id 41 bản ngắn dùng chính sách đã bị thay thế. Trường hợp duy nhất answer ngắn hơn thắng (id 5) là khi answer dài hơn sai. Vì vậy 0.8 chỉ là tín hiệu thống kê; bằng chứng nội dung cho thấy độ dài đi kèm độ chính xác chứ chưa chứng minh judge ưu tiên độ dài. Muốn kiểm tra thật cần cặp đối chứng: cùng nội dung đúng nhưng một bản thêm chữ thừa.

## Calibration đúng model_answer

Helper đánh giá đúng question/model_answer đang có trong fixtures/human_labels_10q.json;
không lấy answer mới của pipeline. Prompt chỉ nhận question, answer, reference.
Không gửi human_label, human_note hoặc phân bố nhãn human vào prompt.

| question_id | model_answer gốc | judge | reasoning (rút gọn) | human_label | agree |
|---|---|---|---|---|---|
| 1 | Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn. | 1 | Đúng số ngày và tính chất có lương | 1 | yes |
| 5 | Cần Giám đốc phòng ban phê duyệt. | 0 | Trên 50 triệu cần Tổng Giám đốc | 0 | yes |
| 12 | Nhân viên được thưởng Tết tối thiểu 1 tháng lương. | 1 | Đúng mức tối thiểu 1 tháng lương | 1 | yes |
| 21 | Được nghỉ 18 ngày phép (15 cơ bản + 3 thâm niên). Lương Senior từ 20 đến 35 triệu. | 1 | Đúng v2024, tính đúng phép và khung lương | 1 | yes |
| 23 | Hoàn trả 25 triệu vì chưa đủ 1 năm cam kết. | 1 | Đúng số tiền hoàn trả và lý do | 1 | yes |
| 29 | Trưởng phòng phê duyệt. Phạt 2% tháng trên 8 triệu. | 0 | Thiếu Kế toán trưởng, tính sai phí phạt | 0 | yes |
| 33 | Phép: 19 ngày. Phụ cấp: 1.500.000 VNĐ/tháng. | 1 | Đúng tổng phụ cấp và số ngày phép | 1 | yes |
| 41 | Nhân viên được nghỉ 12 ngày phép năm. | 0 | Dùng v2023 đã bị thay thế | 0 | yes |
| 46 | Nhân viên thử việc không được nghỉ phép năm và phải xin nghỉ không lương nếu cần. | 1 | Đúng phủ định và phương án thay thế | 1 | yes |
| 50 | Được, miễn là đảm bảo kết nối an toàn. | 0 | Chính sách cấm VPN cá nhân | 0 | yes |

Cohen’s κ: 1.0.
Observed/chance agreement và diễn giải: p_o = 10/10 = 1.0; judge có 6 nhãn 1 và 4 nhãn 0, human cũng 6/4 nên p_e = 0.6·0.6 + 0.4·0.4 = 0.52; κ = (1.0 − 0.52)/(1 − 0.52) = 1.0 — "almost perfect" theo Landis–Koch, vượt ngưỡng bonus 0.6.

## Giới hạn và cách dùng judge

- Mẫu rất nhỏ (5 cặp, 10 nhãn): κ = 1.0 trên 10 câu không đảm bảo judge đồng thuận với người trên phân phối rộng hơn; khoảng tin cậy của κ với n = 10 rất rộng. Các câu trong bộ nhãn khá rõ đúng/sai (sai số liệu, sai phiên bản, sai phủ định), là trường hợp dễ cho judge.
- Có điểm lệch tiêu chuẩn giữa hai quy trình: ở pairwise id 12 judge trừ điểm B (0.7) vì "bỏ điều kiện", nhưng binary judge vẫn cho cùng answer đó nhãn 1. Hai câu hỏi khác nhau (so sánh tương đối và đúng/sai tuyệt đối) nên không chuyển winner A/B thành nhãn 0/1.
- Judge là model lite (`gemini-3.1-flash-lite-preview`) trên free tier; đổi model/prompt cần đo lại κ và position bias trước khi dùng làm gate.
- Cách dùng đề xuất: dùng swap-and-average cho mọi so sánh A/B, coi cặp bất nhất là tie; định kỳ lấy mẫu 20–50 câu có nhãn người để đo lại κ; không coi judge là đáp án tuyệt đối mà là bộ lọc để người xem lại các case lệch.
