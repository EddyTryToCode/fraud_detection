"""
Script tạo 3 notebook EDA cho Tuần 1:
1. notebooks/01_eda_base.ipynb
2. notebooks/02_eda_variants.ipynb
3. notebooks/03_eda_fairness.ipynb
"""

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


# ==========================================
# Notebook 1: 01_eda_base.ipynb
# ==========================================
nb1_cells = [
    make_md_cell("""# 🔍 Notebook 01: EDA Cơ Bản Trên BAF Base Dataset

> **Dự án**: Fraud Detection System — BAF Benchmark (NeurIPS 2022)  
> **Mục tiêu**: Phân tích tổng quan dữ liệu `Base.csv`, kiểm tra tỷ lệ imbalanced, phân phối feature, missing values (sentinel values như `-1`), và mối tương quan giữa các đặc trưng với nhãn `fraud_bool`."""),
    
    make_code_cell("""import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config seaborn style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

PROJECT_ROOT = Path("..").resolve()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
BASE_PATH = RAW_DATA_DIR / "Base.csv"

print(f"Loading data from: {BASE_PATH}")
df = pd.read_csv(BASE_PATH)
print(f"Shape: {df.shape[0]:,} dòng x {df.shape[1]} cột")"""),

    make_md_cell("""## 1. Kiểm tra cấu trúc dữ liệu & Tỷ lệ Class Imbalance"""),
    
    make_code_cell("""print("--- Thông tin Data Types ---")
print(df.dtypes.value_counts())

print("\n--- Kiểm tra Class Imbalance (fraud_bool) ---")
fraud_counts = df['fraud_bool'].value_counts()
fraud_rates = df['fraud_bool'].value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Số lượng': fraud_counts,
    'Tỷ lệ (%)': fraud_rates.round(4)
})
summary_df.index = ['Legitimate (0)', 'Fraud (1)']
print(summary_df)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x='fraud_bool', ax=ax[0], palette=['#2ecc71', '#e74c3c'])
ax[0].set_title('Số lượng case Fraud vs Legitimate')
ax[0].set_xticklabels(['Legitimate (0)', 'Fraud (1)'])

for p in ax[0].patches:
    ax[0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')

ax[1].pie(fraud_counts, labels=['Legitimate (98.9%)', 'Fraud (1.1%)'], 
           autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'], explode=(0, 0.15), startangle=90)
ax[1].set_title('Tỷ lệ Fraud (Class Imbalance Nghiêm Trọng)')
plt.tight_layout()
plt.show()"""),

    make_md_cell("""## 2. Kiểm tra Sentinel Values (-1) làm Missing Values
Trong benchmark BAF, các giá trị thiếu (missing data) được mã hóa dưới dạng giá trị sentinel `-1` ở các cột như `prev_address_months_count` và `bank_months_count`."""),

    make_code_cell("""sentinel_cols = ['prev_address_months_count', 'bank_months_count', 'current_address_months_count']

for col in sentinel_cols:
    if col in df.columns:
        neg_count = (df[col] == -1).sum()
        pct = (neg_count / len(df)) * 100
        print(f"Cột '{col}': {neg_count:,} giá trị = -1 ({pct:.2f}%)")"""),

    make_md_cell("""## 3. Phân phối các Feature Số Quan Trọng (Numerical Features)"""),

    make_code_cell("""num_cols = ['income', 'name_email_similarity', 'credit_risk_score', 
            'proposed_credit_limit', 'session_length_in_minutes', 'velocity_24h']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='fraud_bool', kde=True, ax=axes[i], 
                 bins=30, palette=['#2ecc71', '#e74c3c'], stat="density", common_norm=False)
    axes[i].set_title(f'Phân phối: {col}')

plt.tight_layout()
plt.show()"""),

    make_md_cell("""## 4. Phân tích Feature Phân Loại (Categorical Features)"""),

    make_code_cell("""cat_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    fraud_by_cat = df.groupby(col)['fraud_bool'].agg(['count', 'mean']).reset_index()
    fraud_by_cat['fraud_rate_%'] = fraud_by_cat['mean'] * 100
    
    sns.barplot(data=fraud_by_cat, x=col, y='fraud_rate_%', ax=axes[i], palette='Reds_r')
    axes[i].set_title(f'Tỷ lệ Fraud (%) theo {col}')
    axes[i].set_ylabel('Fraud Rate (%)')
    axes[i].tick_params(axis='x', rotation=30)

# Clear last unused subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()"""),

    make_md_cell("""## 5. Ma Trận Tương Quan (Correlation Matrix với fraud_bool)"""),

    make_code_cell("""numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fraud_corr = corr['fraud_bool'].sort_values(ascending=False)

print("--- Top tương quan với fraud_bool ---")
print(fraud_corr)

plt.figure(figsize=(8, 10))
sns.barplot(x=fraud_corr.values[1:], y=fraud_corr.index[1:], palette='vlag')
plt.title('Tương quan Pearson giữa các Feature Số với fraud_bool')
plt.xlabel('Hệ số tương quan')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()""")
]

# ==========================================
# Notebook 2: 02_eda_variants.ipynb
# ==========================================
nb2_cells = [
    make_md_cell("""# 🧪 Notebook 02: So Sánh 6 Variant Benchmark BAF (NeurIPS 2022)

> **Mục tiêu**: Phân tích sự khác biệt giữa 6 variant (`Base.csv`, `Variant I.csv` ... `Variant V.csv`), đánh giá **temporal dynamics** (biến `month`) và **distribution shifts** giữa các biến thể dữ liệu."""),

    make_code_cell("""import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
PROJECT_ROOT = Path("..").resolve()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

variants = ["Base.csv", "Variant I.csv", "Variant II.csv", "Variant III.csv", "Variant IV.csv", "Variant V.csv"]
variant_dfs = {}

for v in variants:
    path = RAW_DATA_DIR / v
    if path.exists():
        print(f"Loading {v} ...")
        variant_dfs[v] = pd.read_csv(path)
    else:
        print(f"⚠️ Warning: {v} not found!")

print(f"\nĐã load {len(variant_dfs)} variants.")"""),

    make_md_cell("""## 1. So Sánh Quy Mô Dữ Liệu & Tỷ Lệ Fraud Giữa Các Variant"""),

    make_code_cell("""stats = []
for name, vdf in variant_dfs.items():
    n_rows = len(vdf)
    n_fraud = vdf['fraud_bool'].sum()
    fraud_rate = (n_fraud / n_rows) * 100
    stats.append({
        'Variant': name.replace('.csv', ''),
        'Số dòng': f"{n_rows:,}",
        'Số ca Fraud': f"{n_fraud:,}",
        'Tỷ lệ Fraud (%)': round(fraud_rate, 3)
    })

stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=stats_df, x='Variant', y='Tỷ lệ Fraud (%)', palette='Reds_d')
plt.title('Tỷ Lệ Fraud (%) Giữa Các Variant BAF Benchmark')
plt.ylim(0, 2.0)
for i, row in stats_df.iterrows():
    plt.text(i, row['Tỷ lệ Fraud (%)'] + 0.03, f"{row['Tỷ lệ Fraud (%)']}%", ha='center', fontweight='bold')
plt.show()"""),

    make_md_cell("""## 2. Temporal Dynamics: Fraud Rate Theo Thời Gian (`month` 0 đến 7)"""),

    make_code_cell("""base_df = variant_dfs["Base.csv"]
monthly_stats = base_df.groupby('month')['fraud_bool'].agg(['count', 'sum', 'mean']).reset_index()
monthly_stats['fraud_rate_%'] = monthly_stats['mean'] * 100

fig, ax1 = plt.subplots(figsize=(12, 5))

color = 'tab:blue'
ax1.set_xlabel('Month (0 = Tháng đầu tiên, 7 = Tháng cuối)')
ax1.set_ylabel('Tổng số giao dịch', color=color)
ax1.bar(monthly_stats['month'], monthly_stats['count'], color=color, alpha=0.4, label='Số giao dịch')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Tỷ lệ Fraud (%)', color=color)
ax2.plot(monthly_stats['month'], monthly_stats['fraud_rate_%'], color=color, marker='o', linewidth=2.5, label='Fraud Rate (%)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Biến Đổi Temporal (Concept Drift) Của Fraud Rate Qua Các Tháng (Base.csv)')
plt.grid(False)
plt.show()"""),

    make_md_cell("""## 3. So Sánh Bias Nhóm Tuổi (`customer_age`) Giữa Base vs Variant I vs Variant II"""),

    make_code_cell("""comparison_variants = ["Base.csv", "Variant I.csv", "Variant II.csv"]

plt.figure(figsize=(12, 6))
for v_name in comparison_variants:
    if v_name in variant_dfs:
        v_df = variant_dfs[v_name]
        sns.kdeplot(v_df['customer_age'], label=v_name.replace('.csv', ''), linewidth=2)

plt.title('Distribution Shift: Phân Phối Độ Tuổi Khách Hàng (customer_age) Giữa Các Variant')
plt.xlabel('Customer Age')
plt.ylabel('Mật độ (Density)')
plt.legend()
plt.show()""")
]

# ==========================================
# Notebook 3: 03_eda_fairness.ipynb
# ==========================================
nb3_cells = [
    make_md_cell("""# ⚖️ Notebook 03: Phân Tích Thuộc Tính Nhạy Cảm & Fairness Baseline

> **Mục tiêu**: Phân tích thuộc tính nhân khẩu học nhạy cảm (`customer_age`, `employment_status`, `income`) để đặt nền móng cho đánh giá tính công bằng (**Fairness-aware ML**) ở Tuần 5."""),

    make_code_cell("""import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
PROJECT_ROOT = Path("..").resolve()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
BASE_PATH = RAW_DATA_DIR / "Base.csv"

df = pd.read_csv(BASE_PATH)
print(f"Base.csv loaded: {len(df):,} rows")"""),

    make_md_cell("""## 1. Phân Nhóm Độ Tuổi (Age Bins) & Fraud Rate"""),

    make_code_cell("""age_bins = [0, 25, 40, 60, 100]
age_labels = ['<25 (Trẻ)', '25-40 (Trung niên trẻ)', '40-60 (Trung niên)', '>60 (Cao tuổi)']
df['age_group'] = pd.cut(df['customer_age'], bins=age_bins, labels=age_labels)

age_fairness = df.groupby('age_group')['fraud_bool'].agg(['count', 'sum', 'mean']).reset_index()
age_fairness['fraud_rate_%'] = age_fairness['mean'] * 100

print("--- Fraud Rate Theo Nhóm Tuổi ---")
print(age_fairness)

plt.figure(figsize=(10, 5))
sns.barplot(data=age_fairness, x='age_group', y='fraud_rate_%', palette='viridis')
plt.title('Tỷ Lệ Gian Lận (%) Theo Nhóm Tuổi (Protected Attribute: Age)')
plt.ylabel('Fraud Rate (%)')
for i, row in age_fairness.iterrows():
    plt.text(i, row['fraud_rate_%'] + 0.02, f"{row['fraud_rate_%']:.2f}%", ha='center', fontweight='bold')
plt.show()"""),

    make_md_cell("""## 2. Phân Tích Tình Trạng Việc Làm (`employment_status`)"""),

    make_code_cell("""emp_fairness = df.groupby('employment_status')['fraud_bool'].agg(['count', 'sum', 'mean']).reset_index()
emp_fairness['fraud_rate_%'] = emp_fairness['mean'] * 100
emp_fairness = emp_fairness.sort_values(by='fraud_rate_%', ascending=False)

print("--- Fraud Rate Theo Tình Trạng Việc Làm ---")
print(emp_fairness)

plt.figure(figsize=(10, 5))
sns.barplot(data=emp_fairness, x='employment_status', y='fraud_rate_%', palette='rocket')
plt.title('Tỷ Lệ Gian Lận (%) Theo Employment Status')
plt.ylabel('Fraud Rate (%)')
plt.show()"""),

    make_md_cell("""## 3. Phân Tích Mức Thu Nhập (`income`)"""),

    make_code_cell("""inc_fairness = df.groupby('income')['fraud_bool'].agg(['count', 'sum', 'mean']).reset_index()
inc_fairness['fraud_rate_%'] = inc_fairness['mean'] * 100

plt.figure(figsize=(10, 5))
sns.lineplot(data=inc_fairness, x='income', y='fraud_rate_%', marker='o', color='purple', linewidth=2.5)
plt.title('Xu Hướng Fraud Rate (%) Theo Mức Thu Nhập (Income Bracket)')
plt.xlabel('Income Level (0.1 - 0.9)')
plt.ylabel('Fraud Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()"""),

    make_md_cell("""## 4. Tổng Kết Phân Tích Fairness Cho Tuần 5
1. **Độ tuổi (`customer_age`)**: Có sự chênh lệch rõ rệt về tỷ lệ fraud thực tế giữa các nhóm tuổi.
2. **Tình trạng việc làm (`employment_status`)**: Một số nhóm việc làm có tỷ lệ gian lận cao hơn hẳn mặt bằng chung (1.1%).
3. **Thu nhập (`income`)**: Fraud rate thay đổi theo hạn mức thu nhập.
-> Đây sẽ là các protected attributes chính để đánh giá **Equal Opportunity Difference** và **Demographic Parity** ở **Tuần 5**.""")
]


if __name__ == "__main__":
    create_notebook(nb1_cells, NOTEBOOKS_DIR / "01_eda_base.ipynb")
    create_notebook(nb2_cells, NOTEBOOKS_DIR / "02_eda_variants.ipynb")
    create_notebook(nb3_cells, NOTEBOOKS_DIR / "03_eda_fairness.ipynb")
