"""
Script tự động train và đánh giá 3 model: Logistic Regression, XGBoost, LightGBM.
Lưu lại model tốt nhất và báo cáo kết quả đánh giá (metrics).
"""

import sys
import json
import joblib
from pathlib import Path
import pandas as pd
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_processed_data
from src.imbalance import calculate_scale_pos_weight
from src.models.logistic_baseline import build_logistic_baseline
from src.models.xgboost_model import build_xgboost_model
from src.models.lightgbm_model import build_lightgbm_model
from src.evaluation import evaluate_model, compare_models

def main():
    print("🚀 STARTING MODEL TRAINING & EVALUATION...")
    
    # 1. Load Parquet Data
    try:
        df_train = load_processed_data("train.parquet")
        df_test = load_processed_data("test.parquet")
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy dữ liệu processed. Hãy chạy scripts/run_pipeline.py trước.")
        sys.exit(1)
        
    X_train = df_train.drop(columns=['fraud_bool', 'month'], errors='ignore')
    y_train = df_train['fraud_bool'].values
    
    X_test = df_test.drop(columns=['fraud_bool', 'month'], errors='ignore')
    y_test = df_test['fraud_bool'].values
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 2. Xây dựng model
    spw = calculate_scale_pos_weight(y_train)
    models = {
        'Logistic Regression (Baseline)': build_logistic_baseline(),
        'XGBoost': build_xgboost_model(scale_pos_weight=spw),
        'LightGBM': build_lightgbm_model(scale_pos_weight=spw)
    }
    
    results = {}
    trained_models = {}
    
    # 3. Train & Evaluate
    for name, model in models.items():
        print(f"\n⏳ Đang huấn luyện: {name} ...")
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        # LightGBM và XGBoost predict_proba trả về [P(0), P(1)]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Đánh giá (ngưỡng 0.5 mặc định)
        metrics = evaluate_model(y_test, y_pred_proba, threshold=0.5)
        metrics['Train Time (s)'] = round(train_time, 2)
        
        results[name] = metrics
        trained_models[name] = model
        print(f"✅ Hoàn thành {name} (AUROC: {metrics['AUROC']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f})")

    # 4. So sánh kết quả
    print("\n📊 BẢNG SO SÁNH CÁC MODEL TRÊN TẬP TEST:")
    df_results = compare_models(results)
    print(df_results.to_string())
    
    # 5. Lưu kết quả và model tốt nhất
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Lưu bảng kết quả
    df_results.to_csv(models_dir / "experiment_results.csv")
    
    # Lấy model có AUROC cao nhất
    best_model_name = df_results.index[0]
    best_model = trained_models[best_model_name]
    print(f"\n🏆 Model tốt nhất: {best_model_name} với AUROC = {df_results.iloc[0]['AUROC']:.4f}")
    
    if 'XGBoost' in best_model_name:
        best_model.save_model(models_dir / "xgboost_best.json")
        print("Đã lưu mô hình XGBoost tại: models/xgboost_best.json")
    else:
        joblib.dump(best_model, models_dir / "best_model.joblib")
        print("Đã lưu mô hình tại: models/best_model.joblib")
        
    print("\n✅ HOÀN TẤT TUẦN 3-4!")

if __name__ == "__main__":
    main()
