# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603993
- **Họ và tên:** CAO TRUNG HIẾU
- **SEED** (in ở ô đầu notebook): 5148

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

| Cách tiếp cận | F1 (test) | F1 (SUB 40) | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.896 | 0.788 | 0.186 | 0.000 |
| 2. BoW + MLP | 0.927 | 0.889 | 0.415 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.914 | 0.974 | 64.186 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.952 | 0.950 | 59.442 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.967 | 0.950 | 38.723 | 0.000 |
| 4a. LLM nhỏ zero-shot (Gemini) | 1.000 | 1.000 | 1013.775 | 0.022 |
| 4b. LLM lớn zero-shot (Gemini) | 1.000 | 1.000 | 2795.507 | 0.051 |
| 4c. LLM nhỏ few-shot (Gemini) | 1.000 | 1.000 | 882.489 | 0.056 |

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

**(a) `ml_lr` — learning rate** (câu hỏi 1.B và Phản ánh cuối câu 3)
- *Tôi đoán:* lr=0.001 thì khả năng các hội tụ của mô hình tới đường dẫn loss quá chậm dẫn tới mô hình chưa tìm được điểm loss thấp nhấp, lr=50 thì bước nhảy quá lớn dẫn tới khả năng mô hình có thể nhảy qua cả điểm cực tiểu loss thấp nhấp hoặc nhảy qua nhảy lại viền thay vì rơi được xuống các điểm hội tụ loss thấp
- *Thực tế và vì sao lệch:*
  - Khi lr=0.001 thì loss bắt đầu giảm gần như tuyến tính, nhưng giá trị loss vẫn rất cao. Cũng khá gần với tôi dự đoán khi nó không tìm tới điểm hội tụ loss thấp nhất
  - Khi lr=50 thì việc dự đoán ban đầu bị sai khá nhiều, nhưng khi tới epoch tầm số 10 thì đột ngột loss nhảy xuống rất thấp và dần hội tụ được. Tức là mô hình ban đầu nhảy bật khỏi điểm hội tụ, nhưng sau cùng lại lọt xuống được điểm hội tụ. Nhưng tôi nghi ngờ ở đây mô hình bắt đầu có hiện tượng overfitting. Kết quả khác với dự đoán của tôi
  - Điều tôi dự đoán sai là lr=50 sẽ khiến cho loss dao động ngoài điểm cực tiểu, nhưng thực tế loss vọt lên xong ngay lập tức hạ xuống dốc cực tiểu. Có thể logistic không phải lúc nào chọn lr=50 cũng tệ

**(b) `tradeoff` — đường Pareto** (câu hỏi 5.1)
- *Tôi đoán:* Trên đường Paerto khả năng sẽ có BoW+LogReg, BoW + MLP, Encoder + MLP head và LLM nhỏ. Bên phải vạch 50ms là Encoder+KKN và các LLM lớn còn lại, do thời gian phản hồi chậm
- *Thực tế và vì sao lệch:* Dự đoán của tôi gần đúng trừ EncoderKNN. Ví dụ bị lấn áp là LLM lớn fewshot bị LLM nhỏ fewshot lấn át, do cả 2 lỗi bằng 0 nhưng 4c nhanh hơn rất nhiều.

**(c) `hard_eval` — 50 case khó** (câu hỏi 6.2 và 6.3)
- *Tôi đoán:* Khả năng accuracy giảm mạnh khi dùng model sử dụng BoW, các model khác bắt đầu tăng dần accuracy lên, LLM sẽ có khả năng kiểm định spam tốt nhất. Nhóm khó nhất với BoW chắc là nhóm né từ khóa. Nhóm khó nhất với LLM là những tin nhắn chứa lệnh ignore instructions nhắm vào chính LLM
- *Thực tế và vì sao lệch:*
  - Nhóm khó nhất với BoW là nhóm thao túng/lừa đảo, spam giọng tự nhiên và né từ khóa do không có từ khóa quen, chữ bị biến dạng, do BoW nhìn văn bản thành các từ rời rạc. Embedding khó nhất ở nhóm thao túng và spam tự nhiên vì nghĩa của chúng giống text bình thường. LLM đọc như người và hiểu bối cảnh, nên gần như khong có nhóm khó.
  - LLM không bị lừa, đúng 100%. Prompt phòng thủ giúp ở chỗ khác: nó sửa tin ham duy nhất mà 4a đoán sai (nhóm "tưởng spam nhưng không phải" từ 90% lên 100%) và không làm hỏng nhóm nào.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

- F1 tụt mạnh nhất ở Encoder+LogReg và Bow+MLP ~0.43. Bow+LogReg và Encoder+MLP cũng giảm ~0.23. LLM gần như không đổi. Điểm test thường "quá lạc quan" vì nó cùng phân phối với train: spam trong đó có từ khoá và mẫu quen thuộc mà mô hình đã thấy, nên mô hình chỉ cần nhận ra mẫu đã biết chứ không cần hiểu. Bộ khó cố tình đưa vào những kiểu mô hình chưa gặp.
- Nhóm khó nhất với BoW là nhóm thao túng/lừa đảo, spam giọng tự nhiên và né từ khóa do không có từ khóa quen, chữ bị biến dạng, do BoW nhìn văn bản thành các từ rời rạc. Embedding khó nhất ở nhóm thao túng và spam tự nhiên vì nghĩa của chúng giống text bình thường. LLM đọc như người và hiểu bối cảnh, nên gần như khong có nhóm khó.

**Hai case cụ thể:**
1. *Thao túng, "Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account?..."*: không có từ khoá spam nào. BoW chỉ cộng trọng số từng từ (nhiều từ đời thường kéo về ham) và không hiểu cả câu là lời mạo danh người thân để xin tiền. LLM hiểu ý đồ nên đoán đúng.
2. *Spam giọng tự nhiên, "Hey it's Jess from the gym ... members-only thing this week, 3 months for the price of 1, just reply JOIN..."*: giọng thân mật như người quen, nên BoW không thấy dấu hiệu spam quen thuộc. Embedding nhìn nghĩa chung của câu cũng dễ coi đây là tin bình thường.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề #1: ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?

Cách làm:
require_prediction("ml_f1")
X_tr, X_va, X_te = vectorize(texts_tr, vocab), vectorize(texts_va, vocab), vectorize(texts_te, vocab)
t0 = time.perf_counter()
w, b, losses = train_logreg(X_tr, y_tr, lr=0.5, epochs=300)
train_time = time.perf_counter() - t0

predict_bow_lr = lambda texts: (predict_proba(vectorize(texts, vocab), w, b) >= 0.2).astype(int)
pred_te, bench = benchmark(predict_bow_lr, texts_te, texts_te[SUB])
m_tr = metrics(y_tr, (predict_proba(X_tr, w, b) >= 0.5).astype(int)); m_te = metrics(y_te, pred_te)
print("TEST :", {k: round(v, 3) for k, v in m_te.items()})

predict_bow_lr2 = lambda texts: (predict_proba(vectorize(texts, vocab), w, b) >= 0.8).astype(int)
pred_te2, bench2 = benchmark(predict_bow_lr2, texts_te, texts_te[SUB])
m_te2 = metrics(y_te, pred_te2)
print("TEST2:", {k: round(v, 3) for k, v in m_te2.items()})

Kết quả & Giải thích:
- Với ngưỡng 0.2: Precision = 0.89, Recall = 0.95, F1 = 0.92
- Với ngưỡng 0.8: Precision = 1, Recall = 0.6, F1 = 0.75
Giải thích: Khi lấy ngưỡng là 0.2 thì mô hình bắt đúng được nhiều trường hợp spam, nhưng độ chính xác giảm do mô hình thích đánh spam cả cho các dữ liệu bình thường. Ngược lại khi ngưỡng là 0.8 thì mô hình bắt được ít trường hợp spam hơn, do một số text BoW bị một số từ thông thường làm áp đảo các từ chứa nội dung spam. Khiến cho tỉ lệ bắt đúng spam giảm và làm F1 giảm theo
- Một ngân hàng sợ chặn nhầm tin OTP (không phải SPAM) thì nên tăng ngưỡng lên vì nếu giảm ngưỡng sẽ có khả năng tăng FP (dự đoán là spam nhưng thật ra không hpari)

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

- Chọn phương án cho sản phẩm thật: Nên xài BoW+LogReg do độ trễ thấp 0,12ms. Nếu muốn độ trễ và chất lượng thì nên chọn Encoder+MLP Head do F1 cao và độ trễ cũng thấp. Nếu xài LLM thì: 0,022 USD cho 1000 tin, vậy mỗi ngày khoảng 220 USD, vậy khoảng 6600 USD/tháng
- Theo tôi nên cài Encoder+MLP để xử lý tin chắc chắn, không chắc thì xài LLM. Nó tốt hơn vì vừa xử lý được các tin có độ chắc chắn cao và LLM thì chỉ cần xử lý lượng ít tin khó.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 0 đến 3 (Phần tự code) | Giải thích khái niệm (data leakage, gradient, learning rate, overfit, cú pháp Linear/ReLU, cosine, kNN); nhờ xem lại code 3.2 và 3.3 | Chỉ giải thích khái niệm và gợi ý cách lập luận, không viết hộ TODO; chỉ ra lỗi lấy idx[:k//2] ở 3.3 | Tự viết các TODO, chạy ô kiểm tra, tự sửa 3.3; so dự đoán với kết quả thật |
| 1 (đoạn in trọng số) | Nhờ viết đoạn code ngắn chỉ ra bag-of-words (x, w, z, p) | AI viết đoạn code chẩn đoán (không phải TODO) | Chạy trên model của tôi, dùng số thật để viết Giải thích 1 |
| 4 (TODO 4.1 đến 4.4) | Nhờ viết call_openai, build_classification_prompt, parse_label, llm_classify_batch | AI viết cả 4 hàm | Dán vào ô, chạy ô thử, chạy thí nghiệm 4.A và 4.B (invalid = 0) |
| 4 (cấu hình Gemini) | Lỗi 429 và lỗi 400 (thinking MINIMAL) | Giải thích nguyên nhân (key free tier, reasoning_effort); đề xuất đổi minimal thành low | Kiểm tra key, sửa PROVIDERS, chạy lại ô cấu hình |
| 5 (dự đoán, câu hỏi, MY_CASES, tính tay, phản ánh cuối) | Nhờ gợi ý dự đoán, nhận xét bảng và biểu đồ, nghĩ case riêng, giải 5 phép tính, viết nháp phản ánh | AI gợi ý, viết bản nháp, nghĩ 9 case, giải tính tay (kiểm bằng hash) | Đối chiếu số với bảng của tôi, sửa dự đoán sai, đọc lại nhãn các case, tự chép bước tính tay ra giấy |
| Thử thách cá nhân (đề #1) | Hỏi phương pháp làm ban đầu | AI từ chối gợi ý; chỉ giải thích ô chọn đề, nhắc thiếu ý trả lời về ngân hàng, sửa chính tả | Tự chạy hai ngưỡng và tự viết câu trả lời |
| Báo cáo này (mục 2, 3, 4, 6) | Nhờ lấy số liệu từ day01_lab.ipynb điền vào báo cáo | AI đọc output đã lưu trong notebook, điền bảng kết quả và viết bản nháp các mục 3, 4, 6 | Cần đối chiếu số với notebook của tôi và sửa lại bằng lời của mình |

**Điều tôi hiểu ra hôm nay:** Điều tôi dự đoán sai là lr=50 sẽ khiến cho loss dao động ngoài điểm cực tiểu, nhưng thực tế loss vọt lên xong ngay lập tức hạ xuống dốc cực tiểu. Có thể logistic không phải lúc nào chọn lr=50 cũng tệ
