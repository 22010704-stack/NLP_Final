# Label Guideline – Phản Hồi Sinh Viên về Môn Học

**Dự án**: Student Feedback Analysis with Deep Learning  
**Phiên bản**: 1.0 | **Ngày**: 02/03/2026

---

## 1. Tổng quan bài toán

Phân loại phản hồi (đánh giá) của sinh viên về môn học vào **3 nhãn**:

| Nhãn | Ký hiệu | Mô tả |
|------|---------|-------|
| **Tích cực** | `positive` (2) | Phản hồi thể hiện sự hài lòng, đánh giá cao |
| **Trung tính** | `neutral` (1) | Phản hồi trung lập, vừa tốt vừa chưa tốt, hoặc chỉ mô tả sự kiện |
| **Tiêu cực** | `negative` (0) | Phản hồi thể hiện sự không hài lòng, phê bình |

---

## 2. Định nghĩa chi tiết từng nhãn

### 2.1 Tích cực (Positive – Label 2)

**Khi nào gán nhãn Positive?**
- Văn bản thể hiện **sự hài lòng rõ ràng** về giảng viên, nội dung, hoặc cách tổ chức môn học.
- Sử dụng từ ngữ tích cực: *nhiệt tình, bổ ích, dễ hiểu, hay, xuất sắc, hài lòng, tận tâm, hiệu quả*.
- Có thể có **một vài ý kiến nhỏ** để cải thiện, nhưng **thái độ chung là tích cực**.

**Ví dụ**:
1. *"Thầy giảng dạy rất nhiệt tình, bài học dễ hiểu và có nhiều ví dụ thực tế."* → ✅ Positive
2. *"Môn học rất bổ ích, tôi học được nhiều kiến thức áp dụng được vào thực tế."* → ✅ Positive
3. *"Cô giáo tận tâm, luôn giải đáp thắc mắc của sinh viên một cách chu đáo."* → ✅ Positive
4. *"Nội dung phong phú, phương pháp dạy hiện đại, thầy có nhiều kinh nghiệm thực tế."* → ✅ Positive
5. *"Tuy còn vài điểm nhỏ cần cải thiện nhưng nhìn chung tôi rất hài lòng với môn học."* → ✅ Positive *(cảm xúc chủ đạo là tích cực)*

---

### 2.2 Trung tính (Neutral – Label 1)

**Khi nào gán nhãn Neutral?**
- Văn bản **không thể hiện cảm xúc rõ ràng**, chỉ mô tả thực tế khách quan.
- Có cả ý kiến tích cực lẫn tiêu cực với **mức độ tương đương nhau** (balanced).
- Đề xuất hoặc yêu cầu mà **không kèm theo đánh giá cảm xúc**.
- Không chắc chắn hoặc **mơ hồ** về thái độ.

**Ví dụ**:
1. *"Môn học ở mức bình thường, nội dung ổn nhưng cần cải thiện thêm một số phần."* → ✅ Neutral
2. *"Thầy dạy được, có một số điểm chưa rõ ràng nhưng nhìn chung là ổn."* → ✅ Neutral
3. *"Có những bài hay và những bài chưa hay, tùy từng chủ đề."* → ✅ Neutral
4. *"Môn học có 3 tín chỉ, thi cuối kỳ chiếm 60%."* → ✅ Neutral *(thuần mô tả)*
5. *"Nên bổ sung thêm bài tập thực hành để sinh viên hiểu sâu hơn."* → ✅ Neutral *(đề xuất trung tính)*

---

### 2.3 Tiêu cực (Negative – Label 0)

**Khi nào gán nhãn Negative?**
- Văn bản thể hiện **sự không hài lòng rõ ràng** về giảng viên, nội dung, hoặc tổ chức môn học.
- Sử dụng từ ngữ tiêu cực: *nhàm chán, khó hiểu, không hiệu quả, thất vọng, tệ, kém, lãng phí*.
- Có thể có **một vài điểm tích cực**, nhưng **thái độ chung là tiêu cực**.

**Ví dụ**:
1. *"Bài giảng quá nhàm chán, giáo viên chỉ đọc slide mà không có giải thích."* → ✅ Negative
2. *"Tôi không học được gì sau cả học kỳ, nội dung quá khó mà thiếu ví dụ."* → ✅ Negative
3. *"Giáo viên không quan tâm đến sinh viên, câu hỏi không được trả lời."* → ✅ Negative
4. *"Chương trình học lỗi thời, không phù hợp với nhu cầu thực tế của ngành."* → ✅ Negative
5. *"Dù có một số điểm hay nhưng nhìn chung môn học này thực sự thất vọng."* → ✅ Negative

---

## 3. Trường hợp khó & cách xử lý

| Trường hợp | Hướng xử lý |
|------------|-------------|
| Văn bản quá ngắn (< 5 từ) | Gán nhãn theo từ ngữ cảm xúc; nếu không có → `OTHER/UNCLEAR` |
| Có cả tích cực và tiêu cực đồng đều | → `neutral` |
| Tiếng Anh hoặc ngôn ngữ khác | Loại khỏi dataset |
| Nội dung không liên quan đến môn học | Loại khỏi dataset |
| Mơ hồ, không xác định được | → `OTHER/UNCLEAR` → loại bỏ |
| Phủ định ("không tệ", "không hẳn là không hay") | Phân tích ngữ nghĩa, đánh giá cảm xúc thực sự |

---

## 4. Quy trình gán nhãn

1. **Đọc toàn bộ** câu phản hồi trước khi gán nhãn.
2. Xác định **cảm xúc chủ đạo** (không chỉ một vài từ khóa).
3. Kiểm tra **phủ định** ("không", "chẳng", "chưa") vì chúng đảo nghĩa câu.
4. Nếu **không chắc**: đánh dấu `UNCLEAR` và chuyển sang mẫu tiếp theo.
5. Gán nhãn độc lập, **không tham khảo người gán nhãn khác** cho đến khi hoàn tất batch.

### Inter-Annotator Agreement Protocol
- Annotator 1 và Annotator 2 gán nhãn **độc lập** cho 20% dữ liệu (420 mẫu).
- Tính **Cohen's Kappa** để đo mức độ đồng thuận.
- Kappa ≥ 0.7 = chấp nhận được; Kappa ≥ 0.8 = tốt.
- Những mẫu bất đồng sẽ được thảo luận và thống nhất bởi cả hai.

---

## 5. Thống kê kỳ vọng

| Nhãn | Số mẫu | Tỷ lệ |
|------|--------|-------|
| Negative (0) | 700 | 33.3% |
| Neutral (1) | 700 | 33.3% |
| Positive (2) | 700 | 33.3% |
| **Tổng** | **2,100** | **100%** |
