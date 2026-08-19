# KẾ HOẠCH HOÀN THIỆN: Fraud Detection Portfolio Project
## Đánh giá hiện trạng + Các bước còn lại để tối ưu cho Banking/E-commerce

---

## 1. Đánh giá hiện trạng — đã làm RẤT tốt, vượt kế hoạch gốc

| Notebook | Đã làm | Chất lượng |
|---|---|---|
| 05 — Model Evaluation | So sánh LogReg/XGBoost/LightGBM, ROC/PR curve, confusion matrix. **XGBoost AUROC 0.8895** — vượt baseline NeurIPS tham chiếu (~0.888) | ✅ Tốt, đúng chuẩn |
| 06/06b — Threshold Tuning | Cost-based threshold (1:50), sensitivity analysis (10:1/50:1/100:1), Version A vs B (fairness through unawareness) | ✅ Rất tốt — đúng tư duy business, hiếm sinh viên làm |
| 07 — Fairness Analysis | Fairlearn MetricFrame theo 4 nhóm tuổi, tính Demographic Parity + Equal Opportunity Difference | ✅ Tốt, đúng chuẩn ngành |
| 08 — Debiasing | ThresholdOptimizer (post-processing, equalized_odds). **Base EOD 0.2769 → Debiased EOD 0.0000** | ⚠️ Kết quả ấn tượng nhưng cần bổ sung phân tích đánh đổi (xem mục 2.1) |
| 09 — SHAP Explainability | Global summary plot + dependence plot cho `customer_age`, phát hiện tương tác age×income | ✅ Tốt, đúng chuẩn explainability |

**Nhận xét chung**: Phần "khoa học" (Data Science core) đã hoàn thiện ở mức cao — đủ để làm nền tảng. Phần còn thiếu chủ yếu là **lớp kỹ sư hóa (Engineering layer)** biến nó từ "notebook nghiên cứu" thành "hệ thống production" — đây chính là phần khiến banking/e-commerce "phê" thật sự, vì họ tuyển AI Engineer, không phải chỉ Data Scientist làm notebook.

---

## 2. Nâng cấp phần đã làm — 3 việc cần bổ sung trước khi coi là "xong"

### 2.1. Debiasing — giải thích rõ đánh đổi, đừng chỉ khoe EOD = 0.0000

EOD giảm về đúng 0 là **kỳ vọng đúng** của ThresholdOptimizer (nó giải trực tiếp bài toán tối ưu theo constraint này) — không phải phép màu, và nếu chỉ khoe con số này mà không giải thích, người phỏng vấn có kinh nghiệm sẽ hỏi ngay "đánh đổi là gì?".

**Việc cần làm**: thêm 1 bảng so sánh **AUROC/Recall tổng thể trước và sau debiasing** — gần như chắc chắn debiasing làm giảm Recall tổng hoặc tăng False Positives ở 1 vài nhóm để cân bằng nhóm khác. Đây mới là insight thật, đúng tinh thần "không có gì miễn phí" (no free lunch) trong fairness — thêm đoạn markdown phân tích rõ trade-off này vào Notebook 08.

### 2.2. So sánh thêm 1 kỹ thuật debiasing khác — để không chỉ có "1 công cụ"

Hiện tại chỉ có Post-processing (ThresholdOptimizer). Thêm 1 kỹ thuật khác nhóm để câu chuyện "tôi hiểu cả hệ sinh thái fairness" mạnh hơn:
- **Reweighting** (pre-processing, Fairlearn `ExponentiatedGradient` hoặc đơn giản là sample weight theo nhóm) — dễ làm nhất, thêm ~30 dòng code.

Không cần làm 3-4 kỹ thuật như bản kế hoạch học thuật cũ (đã bỏ vì không làm thesis) — chỉ cần **2 kỹ thuật** (đã có Post-processing + thêm Pre-processing) là đủ để tránh bị hỏi "sao chỉ thử 1 cách".

### 2.3. SHAP — thêm Local Explanation (giải thích 1 case cụ thể)

Hiện tại chỉ có Global (summary plot) + Dependence plot (vẫn là phân tích toàn cục). Thiếu **Local Explanation** — giải thích **1 giao dịch cụ thể bị flag** (SHAP force plot/waterfall cho 1 sample). Đây là phần **doanh nghiệp cần nhất thực tế** (nhân viên compliance không cần biết "trung bình model học gì", họ cần biết "tại sao CASE NÀY bị flag"). Thêm 1 section nhỏ vào Notebook 09.

---

## 3. Phần còn thiếu — đây là nơi tạo khác biệt lớn nhất cho CV

### 🎯 Ưu tiên 1: LLM Explanation Layer (biến SHAP kỹ thuật thành giải thích tiếng Việt tự nhiên)

**Notebook 10 mới**: Dùng LangGraph (đã có kinh nghiệm) để chuyển output SHAP (dạng số/feature name) thành 1 đoạn giải thích tiếng Việt tự nhiên cho nhân viên compliance không rành kỹ thuật.

```python
# Input: SHAP values của 1 case + feature names + values thật
# Output: "Giao dịch này bị đánh dấu rủi ro cao chủ yếu do: (1) tài khoản 
#          mới mở dưới 24h, (2) thu nhập khai báo thấp bất thường so với 
#          hạn mức giao dịch, (3) email có độ tương đồng thấp với tên khách hàng..."
```

**Vì sao đây là ưu tiên 1**: đây chính là mảnh ghép "AI Engineer" (kết hợp LLM + ML truyền thống) mà banking/e-commerce đang tìm — không công ty nào chỉ cần Data Scientist thuần túy nữa.

### 🎯 Ưu tiên 2: Production Serving (FastAPI + Docker)

**Notebook/script mới → chuyển thành app thật**:
```
src/api/
  ├── main.py          # FastAPI app
  ├── schemas.py        # Pydantic request/response models
  └── inference.py      # Load model + SHAP + LLM explanation, trả về JSON

Endpoint: POST /predict
Input: {transaction features...}
Output: {
  "risk_score": 0.73,
  "decision": "flag_for_review",
  "top_reasons": [...],
  "explanation_vi": "Giao dịch này bị đánh dấu vì..."
}
```
Đóng gói bằng Docker (`Dockerfile` + `docker-compose.yml`) — đây chính là điểm khiến CV "chạy được thật", không chỉ "trong notebook".

### 🎯 Ưu tiên 3: Monitoring Dashboard (Streamlit) + Drift Detection

Tận dụng đúng cấu trúc temporal (`month`) đã có sẵn — dashboard đơn giản:
- Biểu đồ AUROC/Recall theo từng tháng (giả lập model chạy qua thời gian)
- Cảnh báo khi phân phối feature quan trọng (theo SHAP) dịch chuyển đáng kể (drift alert đơn giản: so sánh phân phối tháng hiện tại vs tháng train)

### 🎯 Ưu tiên 4 (bonus, không bắt buộc): Nhánh e-commerce (IEEE-CIS)

Đúng như đã bàn — chỉ làm nếu còn thời gian sau 3 ưu tiên trên, chứng minh pipeline generalize được sang domain khác.

---

## 4. Timeline đề xuất cho phần còn lại (4-6 tuần)

| Tuần | Nội dung |
|---|---|
| 1 | Hoàn thiện mục 2 (nâng cấp Debiasing + SHAP local explanation) |
| 2 | Notebook 10 — LLM Explanation Layer (LangGraph) |
| 3 | FastAPI service — endpoint `/predict`, test bằng Postman/curl |
| 4 | Docker hóa, viết `docker-compose.yml`, test chạy container sạch |
| 5 | Streamlit dashboard + drift monitoring đơn giản |
| 6 | (Nếu còn thời gian) Nhánh e-commerce IEEE-CIS + viết README/case study cuối cùng |

---

## 5. Cập nhật Deliverable Checklist

- [x] Model training + so sánh (Notebook 05)
- [x] Cost-based threshold tuning + sensitivity analysis (Notebook 06/06b)
- [x] Fairness evaluation (Notebook 07)
- [x] Debiasing — cần bổ sung trade-off + 1 kỹ thuật nữa (Notebook 08)
- [x] SHAP global — cần bổ sung local explanation (Notebook 09)
- [ ] **LLM Explanation Layer tiếng Việt** (Notebook 10 — chưa làm)
- [ ] **FastAPI serving** (chưa làm)
- [ ] **Docker packaging** (chưa làm)
- [ ] **Streamlit monitoring dashboard** (chưa làm)
- [ ] Demo link public (Streamlit Cloud/HF Spaces)
- [ ] README/case study tổng hợp toàn bộ pipeline

---

## 6. Câu CV sau khi hoàn thiện đủ 4 ưu tiên

*"Built and deployed an end-to-end fraud detection system on the NeurIPS 2022 Bank Account Fraud benchmark (XGBoost, AUROC 0.89), including cost-sensitive threshold optimization, fairness evaluation and bias mitigation (Fairlearn, reducing Equal Opportunity Difference from 0.28 to nearly 0), SHAP-based and LLM-generated natural-language explanations, and a containerized FastAPI serving layer with drift monitoring dashboard."*

Đây là câu CV đầy đủ nhất, thể hiện cả 4 vai trò: Data Scientist (fairness/threshold), ML Engineer (model benchmark), AI Engineer (LLM explanation), MLOps (FastAPI/Docker/monitoring) — đúng combo "phê đét" bạn muốn.
