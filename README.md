# Phân tích phản hồi sinh viên (Student Feedback Analysis)

Dự án NLP phân loại phản hồi sinh viên về môn học sử dụng Deep Learning.

## 1. Yêu cầu hệ thống
- Python 3.10+
- PyTorch
- Transformers (PhoBERT)
- Underthesea (Vietnamese NLP)
- Scikit-learn, Pandas, Matplotlib, Seaborn

Cài đặt phụ thuộc:
```bash
pip install -r requirements.txt
```

## 2. Cấu trúc thư mục
- `data/`: Chứa các tệp CSV dữ liệu (tự thu thập và public)
- `src/`: Mã nguồn chính
  - `models/`: Chứa 5 kiến trúc mô hình (CNN, RNN, RCNN, Transformer, PhoBERT)
  - `preprocessing.py`: Làm sạch và tiền xử lý văn bản
  - `vocabulary.py`: Xây dựng từ điển cho các mô hình non-BERT
  - `dataset.py`: PyTorch Dataset handlers
  - `trainer.py`: Vòng lặp huấn luyện, early stopping
  - `metrics.py`: Tính toán các chỉ số đánh giá (Accuracy, F1, Matrix)
  - `error_analysis.py`: Phân tích lỗi mô hình
- `Student_Feedback_NLP.ipynb`: Notebook chính chạy toàn bộ thực nghiệm
- `label_guideline.md`: Hướng dẫn gán nhãn dữ liệu

## 3. Cách chạy
1. Tạo dữ liệu tự thu thập:
   ```bash
   python generate_dataset.py
   ```
2. Mở và chạy notebook `Student_Feedback_NLP.ipynb` để:
   - Thống kê dữ liệu và tính Inter-Annotator Agreement (IAA)
   - Tiền xử lý dữ liệu
   - Huấn luyện và đánh giá 5 mô hình
   - So sánh kết quả In-domain, Public và Cross-domain
   - Phân tích lỗi và tính Robustness

## 4. Mô hình triển khai
1. **KimCNN**: Convolutional Neural Network cho văn bản với nhiều kích thước kernel.
2. **BiLSTM + Attention**: Recurrent Neural Network hai chiều kết hợp cơ chế chú ý.
3. **RCNN**: Recurrent Convolutional Neural Network kết hợp ưu điểm của cả hai.
4. **Custom Transformer**: Encoder với Multi-head Attention (tự xây dựng).
5. **PhoBERT + Custom Head**: Pre-trained model cho tiếng Việt với Attentive Pooling head.

## 5. Kết quả mong đợi
- Accuracy, Macro-F1, Weighted-F1 cho tất cả mô hình.
- Confusion Matrix và Error Analysis (≥30 lỗi).
- Đánh giá Robustness (với noise và không dấu).
- So sánh thời gian huấn luyện và kích thước mô hình.
