import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10.0"},
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã tạo: {filepath.relative_to(filepath.parent.parent)}")

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = [
    make_md_cell("""# 🤖 Notebook 10: LLM Explanation Layer (Cầu Nối AI Kỹ Thuật & Nghiệp Vụ Ngân Hàng)

> **Mục tiêu**: Biến các chỉ số toán học trừu tượng (SHAP Values, Feature Importances, Risk Probabilities) thành **báo cáo giải trình tiếng Việt tự nhiên, chuẩn nghiệp vụ Ngân hàng** cho các chuyên viên phòng Chống Gian lận (Fraud/Compliance Analysts).
> 
> **Kiến trúc Pipeline**:
> `Raw Transaction JSON` ➡️ `Preprocessor` ➡️ `XGBoost Model` ➡️ `SHAP Engine` ➡️ `LLM Explainer Layer` ➡️ `Báo Cáo Nghiệp Vụ`"""),
    
    make_code_cell("""import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_processed_data, load_raw_data, temporal_split
from src.explainability import calculate_shap_values, get_top_influential_features
from src.llm_explainer import generate_fraud_explanation, rule_based_explainer_vi

print("Loading Data and Models...")
df_test = load_processed_data("test.parquet")
X_test = df_test.drop(columns=['fraud_bool', 'month'], errors='ignore')
y_test = df_test['fraud_bool'].values

estimator = xgb.XGBClassifier()
estimator.load_model(str(PROJECT_ROOT / "models" / "xgboost_best.json"))

# Lấy 1,000 mẫu để tính SHAP
base_value, shap_values, X_sample = calculate_shap_values(estimator, X_test, max_samples=1000)
y_sample = y_test[X_sample.index]

OPTIMAL_THRESHOLD = 0.48
probs_sample = estimator.predict_proba(X_sample)[:, 1]
preds_sample = (probs_sample >= OPTIMAL_THRESHOLD).astype(int)

print(f"Sẵn sàng thử nghiệm trên {len(X_sample)} giao dịch!")"""),

    make_md_cell("""## 1. Tìm Kiếm Các Hồ Sơ Đại Diện (Case Selection)
Chúng ta sẽ trích xuất 3 tình huống nghiệp vụ thực tế:
1. **Case 1 (True Positive)**: Gian lận thực sự bị phát hiện.
2. **Case 2 (False Positive)**: Khách hàng chân chính bị hệ thống nghi ngờ nhầm.
3. **Case 3 (True Negative)**: Khách hàng an toàn, phê duyệt thẳng."""),

    make_code_cell("""tp_idx = np.where((preds_sample == 1) & (y_sample == 1))[0][0]
fp_idx = np.where((preds_sample == 1) & (y_sample == 0))[0][0]
tn_idx = np.where((preds_sample == 0) & (y_sample == 0))[0][0]

print(f"Case 1 (True Positive)  : Index {tp_idx:3d} | Risk Score = {probs_sample[tp_idx]:.2%}")
print(f"Case 2 (False Positive) : Index {fp_idx:3d} | Risk Score = {probs_sample[fp_idx]:.2%}")
print(f"Case 3 (True Negative)  : Index {tn_idx:3d} | Risk Score = {probs_sample[tn_idx]:.2%}")"""),

    make_md_cell("""## 2. Thử Nghiệm Case 1: Phát Hiện Gian Lận Thực Sự (True Positive)"""),

    make_code_cell("""# 1. Trích xuất Top 5 lý do kỹ thuật từ SHAP
top_reasons_tp = get_top_influential_features(
    shap_values_row=shap_values[tp_idx],
    feature_names=list(X_sample.columns),
    feature_values_row=X_sample.iloc[tp_idx].values,
    top_k=5
)

# 2. Sinh báo cáo nghiệp vụ tiếng Việt
explanation_tp = generate_fraud_explanation(
    risk_score=float(probs_sample[tp_idx]),
    threshold=OPTIMAL_THRESHOLD,
    top_reasons=top_reasons_tp,
    use_llm=True # Sẽ dùng Gemini nếu có API key, tự động dùng Domain NLG nếu không có key
)

print(explanation_tp)"""),

    make_md_cell("""## 3. Thử Nghiệm Case 2: Phân Tích Bắt Oan (False Positive)
> Trường hợp này rất quan trọng trong dịch vụ khách hàng: Khi khách hàng gọi lên khiếu nại "Tại sao tôi bị từ chối mở thẻ?", nhân viên tổng đài cần biết chính xác lý do để hỗ trợ gỡ phong tỏa."""),

    make_code_cell("""top_reasons_fp = get_top_influential_features(
    shap_values_row=shap_values[fp_idx],
    feature_names=list(X_sample.columns),
    feature_values_row=X_sample.iloc[fp_idx].values,
    top_k=5
)

explanation_fp = generate_fraud_explanation(
    risk_score=float(probs_sample[fp_idx]),
    threshold=OPTIMAL_THRESHOLD,
    top_reasons=top_reasons_fp,
    use_llm=True
)

print(explanation_fp)"""),

    make_md_cell("""## 4. Thử Nghiệm Case 3: Hồ Sơ An Toàn (True Negative)"""),

    make_code_cell("""top_reasons_tn = get_top_influential_features(
    shap_values_row=shap_values[tn_idx],
    feature_names=list(X_sample.columns),
    feature_values_row=X_sample.iloc[tn_idx].values,
    top_k=5
)

explanation_tn = generate_fraud_explanation(
    risk_score=float(probs_sample[tn_idx]),
    threshold=OPTIMAL_THRESHOLD,
    top_reasons=top_reasons_tn,
    use_llm=True
)

print(explanation_tn)"""),

    make_md_cell("""## 5. Kết Luận Về Giá Trị Kiến Trúc AI Engineer
Sự kết hợp giữa **Machine Learning truyền thống (XGBoost + SHAP)** và **Lớp Ngôn Ngữ Tự Nhiên (NLG/LLM Explainer)** mang lại 3 lợi thế cạnh tranh vượt trội:
1. **Tuân thủ quy định (Regulatory Compliance)**: Ngân hàng Nhà nước và các cơ quan giám sát tài chính (GDPR/Fair Lending) bắt buộc mọi quyết định từ chối tín dụng phải có văn bản giải trình rõ ràng.
2. **Nâng cao năng suất**: Nhân viên vận hành không cần đọc bảng số liệu phức tạp mà có ngay kết luận súc tích trong 3 giây.
3. **Sẵn sàng tích hợp API**: Định dạng JSON từ module này sẽ được nạp trực tiếp vào **FastAPI Serving Endpoint** ở bước tiếp theo!""")
]

if __name__ == "__main__":
    create_notebook(cells, NOTEBOOKS_DIR / "10_llm_explanation.ipynb")
