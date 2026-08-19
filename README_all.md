# Fraud Detection System — BAF Benchmark (NeurIPS 2022)

> Hệ thống phát hiện gian lận mở tài khoản ngân hàng, benchmark chuẩn NeurIPS + Fairness-aware ML + Explainability (SHAP + LLM tiếng Việt). Dự án career-first, có thể rẽ nhánh thành khóa luận.

---

## 📌 Định vị dự án

Dự án được thiết kế để **tối ưu cho tuyển dụng trước, học thuật sau**. Mọi quyết định thiết kế (dataset, kiến trúc, phạm vi) đều trả lời câu hỏi: *"Nhà tuyển dụng banking/fintech/e-commerce nhìn vào sẽ nghĩ gì?"*

**Câu hỏi nghiên cứu chính (career-focused):**
Xây hệ thống phát hiện gian lận mở tài khoản đạt hiệu năng cạnh tranh với baseline NeurIPS, triển khai dưới dạng service production-ready, có khả năng giải thích quyết định và giám sát model drift.

**Câu hỏi mở rộng (kích hoạt nếu rẽ nhánh khóa luận):**
Đánh đổi giữa hiệu năng phát hiện gian lận và tính công bằng (fairness) giữa các nhóm nhân khẩu học thay đổi thế nào khi áp dụng các kỹ thuật cân bằng lớp khác nhau — và liệu lớp giải thích bằng LLM có giúp compliance officer tin tưởng và audit quyết định model tốt hơn không?

---

## 📊 Dataset & Benchmark

**Chính: BAF — Bank Account Fraud Dataset Suite** (Feedzai, NeurIPS 2022 Datasets & Benchmarks Track)

- 6 variant (Base + Variant I-V), mỗi variant ~1 triệu dòng giao dịch mở tài khoản, đã ẩn danh hóa.
- Có yếu tố temporal dynamics + class imbalance nghiêm trọng (~1.1% fraud) — đúng đặc điểm dữ liệu thật.
- Có sẵn thuộc tính nhạy cảm (nhóm tuổi, tình trạng việc làm, % thu nhập) để đánh giá fairness.
- Vẫn là benchmark chuẩn hiện hành cho bài toán này — nhiều paper 2025-2026 vẫn dùng làm dataset chính; chưa có gì thay thế.
- Link: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

**Phụ (mở rộng câu chuyện sang e-commerce): IEEE-CIS Fraud Detection (Kaggle)** — ~590K giao dịch, dùng để chứng minh model generalize được sang domain khác.

### Mốc SOTA hiện tại để đối chiếu (BAF-Base, AUROC)

| Model | AUROC |
|---|---|
| FFN (feed-forward cơ bản) | 0.8676 |
| TabTransformer | 0.8721 |
| FT-Transformer (tốt nhất hiện nay) | 0.8988 |
| LightGBM/XGBoost tối ưu (tham chiếu) | ~0.888 |

→ **Mục tiêu hợp lý**: đạt AUROC 0.88–0.90 với XGBoost/LightGBM là đã cạnh tranh với SOTA, không cần thắng transformer mới nhất.

**⚠️ Đừng chỉ báo cáo AUC.** Vì fraud chỉ ~1.1% dữ liệu, model AUC cao vẫn có thể bắt rất ít ca gian lận thật nếu chọn sai objective. Bắt buộc báo cáo thêm: **recall trên lớp fraud** + **chi phí vận hành** (số case bị flag oan / số case fraud bắt đúng). Đây là trục phân biệt Data Scientist thật với sinh viên chỉ khoe điểm số.

### Về việc dữ liệu tiếng Anh — giữ nguyên, không cố Việt hóa

Đây là dữ liệu dạng bảng (số, category) — khái niệm "tiếng Việt" không áp dụng ở tầng dữ liệu. Không tồn tại benchmark fraud ngân hàng VN công khai (luật bảo mật ngân hàng chặt); tự chế dữ liệu VN sẽ yếu hơn BAF rất nhiều và không ai đối chiếu được.

**"Tính Việt Nam" đặt ở lớp diễn giải, không phải lớp dữ liệu:**
- Lớp LLM explanation sinh giải thích **tiếng Việt tự nhiên** cho nhân viên compliance.
- Phần thảo luận liên hệ bối cảnh quản lý rủi ro của Ngân hàng Nhà nước VN (KYC, chống rửa tiền).
- Phân tích định tính: các loại gian lận phổ biến ở VN (lừa đảo OTP, chiếm đoạt tài khoản) ánh xạ vào feature nào của BAF.

---

## 🏗️ Kiến trúc hệ thống

```
[BAF dataset (6 variants) + IEEE-CIS]
            │
            ▼
[1] Data & Feature Pipeline (PySpark)
    - Xử lý temporal features, encoding, xử lý imbalance
    - Feature store đơn giản (Parquet/GCS)
            │
            ▼
[2] Model Training & Experimentation
    - Baseline: Logistic Regression (mốc dưới)
    - Core: XGBoost / LightGBM (mốc chính, đối chiếu NeurIPS baseline)
    - Nâng cao (optional): TabNet / FT-Transformer
            │
            ▼
[3] Fairness Evaluation Layer
    - Đo hiệu năng tách theo nhóm tuổi/thu nhập
    - Metric: Equal Opportunity Difference, Predictive Equality
            │
            ▼
[4] Explainability Layer
    - SHAP values cho từng prediction
    - LLM layer (LangGraph): chuyển SHAP values thành giải thích tiếng Việt
            │
            ▼
[5] Serving & Monitoring
    - FastAPI endpoint: nhận giao dịch → trả risk score + explanation
    - Docker container hóa
    - Dashboard Streamlit: real-time metrics, drift alert
```

---

## 🎯 Skill mapping — vì sao 1 dự án show được cả 3 vai trò

| Vai trò | Phần nào trong dự án | Câu nói trong phỏng vấn |
|---|---|---|
| **Data Scientist** | Feature engineering, fairness analysis, trade-off performance/fairness | "Tôi đã phân tích trade-off giữa AUC và fairness metric khi áp dụng các kỹ thuật resampling khác nhau" |
| **ML Engineer** | Training pipeline tái lập được, benchmark theo NeurIPS, SHAP | "Tôi đã benchmark model trên bộ dữ liệu NeurIPS 2022, đạt AUC cạnh tranh với baseline gốc" |
| **AI Engineer** | Tích hợp LLM (LangGraph) sinh giải thích tự nhiên | "Tôi đã xây lớp LLM chuyển SHAP values kỹ thuật thành giải thích ngôn ngữ tự nhiên cho stakeholder" |
| **MLOps** | FastAPI, Docker, drift detection | "Hệ thống của tôi phát hiện model drift theo thời gian và cảnh báo tự động" |

---

## 🗓️ Timeline (8-10 tuần)

| Tuần | Nội dung | Checkpoint |
|---|---|---|
| 1 | Setup môi trường, tải BAF + IEEE-CIS, EDA sơ bộ | Hiểu rõ imbalance, temporal shift |
| 2 | Feature pipeline PySpark, xử lý encoding/imbalance | Pipeline chạy end-to-end trên 1 variant |
| 3-4 | Training LogReg → XGBoost/LightGBM, tuning, so sánh NeurIPS baseline | Đạt AUC cạnh tranh (~0.88-0.90) |
| 5 | Fairness evaluation, phân tích trade-off | 🔀 **Checkpoint rẽ nhánh khóa luận** nếu có tín hiệu fairness thú vị |
| 6 | SHAP + LLM explanation layer (LangGraph) | Demo: nhập giao dịch → risk score + giải thích tiếng Việt |
| 7 | FastAPI + Docker packaging | API chạy được, demo qua curl/Postman |
| 8 | Streamlit dashboard + drift monitoring | Dashboard live |
| 9-10 | Polish, viết README/case study, cập nhật CV | Sẵn sàng public repo + demo link |

---

## ✅ Setup — làm theo đúng thứ tự

### Bước 1 — Tạo môi trường conda

```bash
conda create -n fraud python=3.10 -y
conda activate fraud
```

Đầu dòng lệnh sẽ đổi từ `(base)` thành `(fraud)`.

### Bước 2 — Cài thư viện

```bash
cd ~/fraud_detection   # đúng thư mục chứa requirements.txt
pip install -r requirements.txt
```

### Bước 3 — Lấy Kaggle API key

1. https://www.kaggle.com/settings → API → **Create New Token** → tải `kaggle.json`
2. Đặt đúng vị trí:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
3. **Bắt buộc**: vào trang dataset trên web, bấm Download 1 lần để accept điều khoản trước khi dùng API:
   https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

### Bước 4 — Tải dataset

```bash
python scripts/download_data.py
```

### Bước 5 — Kiểm tra dữ liệu

```bash
python scripts/verify_data.py
```

Kỳ vọng: thấy tỷ lệ fraud ~1.1%, danh sách cột đầy đủ (bao gồm `fraud_bool`, các feature như `velocity_24h`, `name_email_similarity`, thuộc tính nhân khẩu học).

---

## 📁 Cấu trúc thư mục

```
fraud-detection-project/
├── data/
│   ├── raw/           ← dữ liệu gốc từ Kaggle (không commit Git)
│   └── processed/     ← dữ liệu sau feature engineering
├── src/                ← code chính (pipeline, model, explainability)
├── scripts/            ← download, verify, train, evaluate
├── notebooks/          ← EDA / thử nghiệm
├── models/              ← model đã train
├── requirements.txt
└── README.md
```

---

## 📦 Deliverable checklist (mục tiêu cuối)

- [ ] GitHub repo public, README có bảng so sánh với NeurIPS baseline
- [ ] Demo link sống (Streamlit Cloud/HuggingFace Spaces)
- [ ] 1 bài viết ngắn mô tả insight về fairness trade-off
- [ ] Dòng CV: *"Built and deployed a fraud detection system on the NeurIPS 2022 Bank Account Fraud benchmark, combining XGBoost with SHAP-based and LLM-generated explanations; evaluated fairness trade-offs across demographic groups"*

---

## 🎓 Nếu rẽ nhánh khóa luận

Nếu ở Tuần 5, fairness evaluation cho thấy bias rõ rệt theo nhóm tuổi/thu nhập, mở rộng thêm:
- So sánh 3-4 kỹ thuật fairness-aware (reweighting, adversarial debiasing, post-processing threshold)
- Case study: giải thích LLM có giúp "compliance officer" giả lập ra quyết định audit tốt hơn không

→ Đề tài: *"Fairness-Aware Fraud Detection with LLM-Assisted Explainability for Compliance Auditing"*

---

## 📚 Tài liệu tham khảo

- Jesus, S. et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.* NeurIPS 2022. https://arxiv.org/abs/2211.13358
- GitHub: https://github.com/feedzai/bank-account-fraud
- IEEE-CIS Fraud Detection: https://www.kaggle.com/c/ieee-fraud-detection
