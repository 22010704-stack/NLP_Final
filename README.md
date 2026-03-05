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
- `generate_dataset.py`: Tạo dữ liệu tự thu thập
- `01_KimCNN.ipynb`: Notebook KimCNN (standalone)
- `02_BiLSTM_Attention.ipynb`: Notebook BiLSTM+Attention (standalone)
- `03_RCNN.ipynb`: Notebook RCNN (standalone)
- `04_Transformer.ipynb`: Notebook Custom Transformer (standalone)
- `05_PhoBERT.ipynb`: Notebook PhoBERT + Attentive Pooling (standalone)
- `label_guideline.md`: Hướng dẫn gán nhãn dữ liệu

## 3. Cách chạy
1. Tạo dữ liệu tự thu thập:
   ```bash
   python generate_dataset.py
   ```
2. Mở và chạy lần lượt các notebook (`01_KimCNN.ipynb`, `02_BiLSTM_Attention.ipynb`, `03_RCNN.ipynb`, `04_Transformer.ipynb`, `05_PhoBERT.ipynb`) để:
   - Tải và tiền xử lý dữ liệu
   - Xây dựng từ điển / Tokenizer
   - Huấn luyện và đánh giá mô hình
   - Hiển thị kết quả (Classification Report, Confusion Matrix, Loss Curve)
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
