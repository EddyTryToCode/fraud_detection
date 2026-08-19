"""
Script thực thi EDA tổng hợp trên dataset BAF (Base.csv & variants)
in ra báo cáo phân tích chi tiết cho Tuần 1.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
BASE_PATH = RAW_DATA_DIR / "Base.csv"

def run_eda():
    print("==================================================")
    print("📊 BÁO CÁO KẾT QUẢ EDA TUẦN 1 — BAF BENCHMARK")
    print("==================================================\n")
    
    if not BASE_PATH.exists():
        print(f"❌ File {BASE_PATH} không tồn tại.")
        sys.exit(1)
        
    print("1. TỔNG QUAN BASE.CSV")
    df_base = pd.read_csv(BASE_PATH)
    print(f"   - Shape: {df_base.shape[0]:,} dòng x {df_base.shape[1]} cột")
    
    fraud_count = df_base['fraud_bool'].sum()
    total_count = len(df_base)
    fraud_rate = (fraud_count / total_count) * 100
    print(f"   - Số ca Fraud (1): {fraud_count:,} / {total_count:,} ({fraud_rate:.2f}%)")
    print(f"   - Số ca Legitimate (0): {total_count - fraud_count:,} ({(100 - fraud_rate):.2f}%)")
    print(f"   - Imbalance Ratio: 1 : {int((total_count - fraud_count)/fraud_count)}")

    print("\n2. PHÂN PHỐI THEO THÁNG (TEMPORAL DRIFT)")
    monthly = df_base.groupby('month')['fraud_bool'].agg(['count', 'sum', 'mean']).reset_index()
    monthly['rate_%'] = monthly['mean'] * 100
    for _, r in monthly.iterrows():
        print(f"   - Month {int(r['month'])}: {int(r['count']):,} giao dịch, {int(r['sum']):,} fraud ({r['rate_%']:.2f}%)")

    print("\n3. GIÁ TRỊ THIẾU SENTINEL (-1)")
    sentinel_cols = ['prev_address_months_count', 'current_address_months_count', 'bank_months_count']
    for col in sentinel_cols:
        if col in df_base.columns:
            n_neg = (df_base[col] == -1).sum()
            print(f"   - {col}: {n_neg:,} dòng = -1 ({(n_neg/total_count)*100:.2f}%)")

    print("\n4. THUỘC TÍNH NHẠY CẢM & FAIRNESS BASELINE")
    # Age groups
    age_bins = [0, 25, 40, 60, 100]
    age_labels = ['<25', '25-40', '40-60', '>60']
    df_base['age_group'] = pd.cut(df_base['customer_age'], bins=age_bins, labels=age_labels)
    age_df = df_base.groupby('age_group', observed=False)['fraud_bool'].agg(['count', 'mean']).reset_index()
    age_df['rate_%'] = age_df['mean'] * 100
    print("   [Tỷ lệ fraud theo Độ Tuổi]")
    for _, r in age_df.iterrows():
        print(f"   - Nhóm tuổi {r['age_group']}: {r['rate_%']:.2f}% (tổng {int(r['count']):,} ca)")

    # Employment status
    emp_df = df_base.groupby('employment_status')['fraud_bool'].agg(['count', 'mean']).reset_index()
    emp_df['rate_%'] = emp_df['mean'] * 100
    emp_df = emp_df.sort_values(by='rate_%', ascending=False)
    print("\n   [Tỷ lệ fraud theo Tình Trạng Việc Làm (Employment Status)]")
    for _, r in emp_df.iterrows():
        print(f"   - Status {r['employment_status']}: {r['rate_%']:.2f}% (tổng {int(r['count']):,} ca)")

    print("\n5. SO SÁNH TỶ LỆ FRAUD GIỮA 6 VARIANTS")
    variants = ["Base.csv", "Variant I.csv", "Variant II.csv", "Variant III.csv", "Variant IV.csv", "Variant V.csv"]
    for v in variants:
        vpath = RAW_DATA_DIR / v
        if vpath.exists():
            vdf = pd.read_csv(vpath, usecols=['fraud_bool'])
            v_fraud = vdf['fraud_bool'].sum()
            v_len = len(vdf)
            print(f"   - {v:15s}: {v_len:,} dòng | {v_fraud:,} fraud ({(v_fraud/v_len)*100:.2f}%)")

    print("\n✅ EDA TUẦN 1 HOÀN THÀNH!")

if __name__ == "__main__":
    run_eda()
