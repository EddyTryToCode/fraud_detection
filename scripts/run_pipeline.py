"""
Chạy toàn bộ Data & Feature Pipeline.
1. Load dữ liệu thô (Base.csv).
2. Split tập Train/Test theo Temporal (month).
3. Apply Feature Engineering (Scaler, Encoder, Missing Indicators).
4. Save thành Parquet file sẵn sàng cho Modeling (Tuần 3).
"""

import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_data, temporal_split, save_processed_data
from src.feature_engineering import apply_feature_engineering
from src.imbalance import calculate_scale_pos_weight

def main():
    print("🚀 STARTING DATA & FEATURE PIPELINE...")
    
    # 1. Load data
    df_raw = load_raw_data("Base.csv")
    
    # 2. Temporal Split
    df_train_raw, df_test_raw = temporal_split(df_raw, train_months=[0,1,2,3,4,5], test_months=[6,7])
    
    # 3. Feature Engineering
    print("\nApplying Feature Engineering on Train set...")
    df_train_processed, preprocessor = apply_feature_engineering(df_train_raw, is_train=True)
    
    print("Applying Feature Engineering on Test set...")
    df_test_processed, _ = apply_feature_engineering(df_test_raw, is_train=False, preprocessor=preprocessor)
    
    # 4. Tính toán scale_pos_weight cho imbalanced data
    y_train = df_train_processed['fraud_bool'].values
    spw = calculate_scale_pos_weight(y_train)
    print(f"\n✅ Suggested scale_pos_weight for XGBoost/LightGBM: {spw:.2f}")
    
    # 5. Save processed data and preprocessor object
    save_processed_data(df_train_processed, "train.parquet")
    save_processed_data(df_test_processed, "test.parquet")
    
    # Lưu preprocessor để dùng lúc serving (Tuần 7)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, joblib_path)
    print(f"Saved feature preprocessor to {joblib_path}")
    
    print("\n✅ PIPELINE CHẠY THÀNH CÔNG! Dữ liệu đã sẵn sàng cho Model Training.")

if __name__ == "__main__":
    main()
