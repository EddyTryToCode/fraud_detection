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

nb4_cells = [
    make_md_cell("""# ⚙️ Notebook 04: Feature Pipeline & Data Preparation

> **Mục tiêu**: Trực quan hóa từng bước trong quá trình xử lý dữ liệu (Tuần 2). Notebook này sẽ giúp bạn nhìn thấy rõ input/output của từng bước thay vì chỉ chạy file `.py` ẩn bên dưới."""),
    
    make_md_cell("""## 1. Load Data & Tách Train/Test (Temporal Split)
Thay vì random split, chúng ta dùng biến `month` để tách tập Train (tháng 0-5) và Test (tháng 6-7)."""),

    make_code_cell("""import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Thêm root path để import module
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_data, temporal_split

# Load data
print("Đang load Base.csv...")
df_raw = load_raw_data("Base.csv")
display(df_raw.head())"""),

    make_code_cell("""# Chia Train/Test
df_train, df_test = temporal_split(df_raw, train_months=[0,1,2,3,4,5], test_months=[6,7])

print(f"Shape Train: {df_train.shape}")
print(f"Shape Test: {df_test.shape}")"""),

    make_md_cell("""## 2. Feature Engineering
Xử lý Missing Values (-1), Scale biến số (StandardScaler) và Encode biến phân loại (OrdinalEncoder)."""),

    make_code_cell("""from src.feature_engineering import apply_feature_engineering

print("Applying Feature Engineering trên tập TRAIN...")
df_train_processed, preprocessor = apply_feature_engineering(df_train, is_train=True)

print("Kích thước sau xử lý:", df_train_processed.shape)
display(df_train_processed.head())"""),

    make_code_cell("""print("Applying Feature Engineering trên tập TEST (dùng chung preprocessor của Train)...")
df_test_processed, _ = apply_feature_engineering(df_test, is_train=False, preprocessor=preprocessor)

print("Kích thước sau xử lý:", df_test_processed.shape)"""),

    make_md_cell("""## 3. Class Imbalance (scale_pos_weight)
Dữ liệu cực kỳ mất cân bằng (fraud ~1.1%). Chúng ta tính trọng số `scale_pos_weight` để gán cho XGBoost/LightGBM ở Tuần 3."""),

    make_code_cell("""from src.imbalance import calculate_scale_pos_weight

y_train = df_train_processed['fraud_bool'].values
spw = calculate_scale_pos_weight(y_train)

print(f"Tỷ lệ số ca âm / số ca dương = {spw:.2f}")
print("-> Ghi nhớ thông số này để config model!")"""),

    make_md_cell("""## 4. Lưu ra định dạng Parquet
Parquet lưu trữ metadata (tên cột, data types) và dung lượng nén nhỏ hơn CSV rất nhiều."""),

    make_code_cell("""from src.data_loader import save_processed_data

# Dữ liệu đã được lưu bởi run_pipeline.py, code này mô phỏng lại
print("Lưu df_train_processed -> data/processed/train.parquet")
print("Lưu df_test_processed -> data/processed/test.parquet")""")
]

if __name__ == "__main__":
    create_notebook(nb4_cells, NOTEBOOKS_DIR / "04_feature_pipeline.ipynb")
