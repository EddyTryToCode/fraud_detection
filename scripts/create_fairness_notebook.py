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
    make_md_cell("""# ⚖️ Notebook 07: Đánh giá độ công bằng (Fairness Evaluation)

> **Mục tiêu**: Phân tích xem mô hình XGBoost tốt nhất của chúng ta có đối xử công bằng với tất cả các nhóm nhân khẩu học hay không.
> **Nhóm nhạy cảm (Sensitive Attribute)**: Dựa vào phân tích EDA (Tuần 1), ta thấy nhóm người cao tuổi (`>60`) có tỷ lệ gian lận tự nhiên cao gấp 8.7 lần nhóm trẻ. Liệu mô hình có "học" được bias này và phạt oan (FP) người cao tuổi nhiều hơn, hoặc bỏ lọt (FN) nhiều hơn không?"""),
    
    make_code_cell("""import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb

sns.set_theme(style="whitegrid")
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_processed_data, load_raw_data, temporal_split
from src.fairness import compute_fairness_metrics

print("Loading Test Data and Model...")
# Lấy file raw để có cột customer_age gốc (vì parquet đã bị scale)
df_raw = load_raw_data("Base.csv")
_, df_test_raw = temporal_split(df_raw)

# Load data đã qua xử lý (để cho vào model)
df_test = load_processed_data("test.parquet")
X_test = df_test.drop(columns=['fraud_bool', 'month'], errors='ignore')
y_test = df_test['fraud_bool'].values

# Load model
model = xgb.XGBClassifier()
model.load_model(str(PROJECT_ROOT / "models" / "xgboost_best.json"))

y_scores = model.predict_proba(X_test)[:, 1]

# Sử dụng Threshold Tối Ưu = 0.48 (Đã tìm ra từ Tuần trước với tỷ lệ cost 1:50)
OPTIMAL_THRESHOLD = 0.48
y_pred = (y_scores >= OPTIMAL_THRESHOLD).astype(int)

print(f"X_test shape: {X_test.shape}")"""),

    make_md_cell("""## 1. Phân chia nhóm tuổi (Age Bins)
Chúng ta sẽ chia `customer_age` thành các nhóm: `<25`, `25-40`, `40-60`, `>60` giống như Tuần 1."""),

    make_code_cell("""# Khôi phục tuổi thật nếu có thể, hoặc dùng file raw
age_series = df_test_raw['customer_age']

bins = [0, 25, 40, 60, 100]
labels = ['<25', '25-40', '40-60', '>60']
age_groups = pd.cut(age_series, bins=bins, labels=labels, right=False)

print(age_groups.value_counts().sort_index())"""),

    make_md_cell("""## 2. Tính toán Fairness Metrics (Bằng Fairlearn)"""),

    make_code_cell("""fairness_results = compute_fairness_metrics(
    y_true=y_test, 
    y_pred=y_pred, 
    y_pred_proba=y_scores, 
    sensitive_features=age_groups
)

mf = fairness_results['metric_frame']
display(mf.by_group)"""),

    make_md_cell("""### 2.1 Các chỉ số Chênh Lệch (Disparities)
- **Demographic Parity Difference**: Sự chênh lệch tỷ lệ giao dịch bị flag là gian lận giữa nhóm cao nhất và thấp nhất.
- **Equal Opportunity Difference (EOD)**: Sự chênh lệch Recall (Tỷ lệ bắt trúng) giữa các nhóm."""),

    make_code_cell("""print(f"Demographic Parity Difference: {fairness_results['dp_diff']:.4f}")
print(f"Equal Opportunity Difference : {fairness_results['eo_diff']:.4f}")

# Cảnh báo rủi ro: Nếu EOD > 0.1, mô hình bị thiên vị (bias) nặng
if fairness_results['eo_diff'] > 0.1:
    print("🚨 CẢNH BÁO: Mô hình đang THIÊN VỊ (Equal Opportunity Difference > 0.1)!")
else:
    print("✅ Mô hình có độ công bằng chấp nhận được (Equal Opportunity Difference <= 0.1).")"""),

    make_md_cell("""## 3. Trực quan hoá (Visualization)"""),

    make_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Recall
mf.by_group['recall'].plot(kind='bar', ax=ax1, color='cornflowerblue')
ax1.set_title('Recall theo Nhóm Tuổi (Ai dễ bị bỏ lọt gian lận hơn?)')
ax1.set_ylim(0, 1.0)
ax1.set_ylabel('Recall')
ax1.tick_params(axis='x', rotation=0)

# Plot Selection Rate
mf.by_group['selection_rate'].plot(kind='bar', ax=ax2, color='salmon')
ax2.set_title('Selection Rate (Nhóm nào bị model gắn cờ gian lận nhiều nhất?)')
ax2.set_ylabel('Tỷ lệ giao dịch bị gắn cờ')
ax2.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()"""),

    make_md_cell("""## 4. Insight Thực Tế (Business Value)
Nhìn vào bảng và biểu đồ trên, ta sẽ kết luận:
1. **Selection Rate**: Nhóm >60 tuổi bị model nghi ngờ nhiều nhất (bị gắn cờ tỷ lệ cao nhất). Tuy nhiên, Tuần 1 chứng minh nhóm này có tỷ lệ fraud tự nhiên rất cao. Do đó đây không phải "cờ oan" hoàn toàn.
2. **Recall (Equal Opportunity)**: Nhóm >60 có Recall cao hay thấp so với `<25`? Nếu chênh lệch EOD > 10%, ngân hàng cần cẩn thận vì model đang bảo vệ nhóm này kém hơn (nếu Recall thấp) hoặc chặn giao dịch quá khắt khe (nếu FPR cao).""")
]

if __name__ == "__main__":
    create_notebook(cells, NOTEBOOKS_DIR / "07_fairness_analysis.ipynb")
