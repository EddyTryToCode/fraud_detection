# Fraud Detection System — BAF Benchmark (NeurIPS 2022)

Dự án phát hiện gian lận mở tài khoản ngân hàng, dùng benchmark **Bank Account Fraud (BAF)** của Feedzai (NeurIPS 2022), với lớp explainability (SHAP + LLM tiếng Việt) và đánh giá fairness.

---

## 🚀 Setup — làm theo đúng thứ tự

### Bước 1 — Tạo Python virtual environment

```bash
# Kiểm tra version Python (khuyến nghị 3.10 hoặc 3.11)
python3 --version

# Tạo venv
python3 -m venv venv

# Kích hoạt
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

Bạn sẽ thấy `(venv)` xuất hiện đầu dòng lệnh — nghĩa là đã vào đúng môi trường ảo.

### Bước 2 — Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Việc này mất khoảng 2-5 phút tùy tốc độ mạng (do XGBoost/LightGBM khá nặng).

### Bước 3 — Lấy Kaggle API key

1. Vào https://www.kaggle.com/settings
2. Kéo xuống mục **API** → bấm **Create New Token**
3. File `kaggle.json` sẽ tự tải về máy
4. Di chuyển file vào đúng vị trí:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

5. **Quan trọng**: vào thẳng trang dataset và bấm nút Download 1 lần trên web để accept điều khoản sử dụng (bắt buộc trước khi dùng API):
   👉 https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

### Bước 4 — Tải dataset

```bash
python scripts/download_data.py
```

Script sẽ tải và giải nén tự động vào `data/raw/`. Dataset gồm 6 file CSV (Base + 5 variant), mỗi file ~1 triệu dòng, tổng dung lượng khoảng 500MB-1GB — tùy tốc độ mạng có thể mất 5-15 phút.

### Bước 5 — Kiểm tra dữ liệu

```bash
python scripts/verify_data.py
```

Script này in ra: số dòng/cột, tỷ lệ fraud/non-fraud, danh sách cột, và vài dòng dữ liệu mẫu — để bạn chắc chắn dữ liệu tải đúng trước khi bắt đầu code chính.

### Bước 6 — Khởi tạo Git (nếu muốn track code ngay từ đầu)

```bash
git init
echo "venv/" >> .gitignore
echo "data/raw/*.csv" >> .gitignore   # dataset lớn, không nên commit lên Git
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
git add .
git commit -m "Initial project setup"
```

---

## 📁 Cấu trúc thư mục

```
fraud-detection-project/
├── data/
│   ├── raw/           ← dữ liệu gốc tải từ Kaggle (không commit lên Git)
│   └── processed/     ← dữ liệu sau feature engineering
├── src/                ← code chính (feature pipeline, model, explainability)
├── scripts/            ← script tiện ích (download, verify, train, evaluate)
├── notebooks/          ← Jupyter notebook để EDA/thử nghiệm
├── models/              ← model đã train (lưu file .pkl/.json)
├── requirements.txt
└── README.md
```

---

## ✅ Sau bước này

Khi `verify_data.py` chạy thành công và bạn thấy được tỷ lệ fraud ~1.1%, bạn đã sẵn sàng cho **Tuần 2: xây feature pipeline** — bước tiếp theo trong kế hoạch.
