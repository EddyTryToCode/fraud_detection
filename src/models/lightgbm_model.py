from lightgbm import LGBMClassifier

def build_lightgbm_model(scale_pos_weight=1.0):
    """
    Tạo mô hình LightGBM cho Fraud Detection.
    - scale_pos_weight: Tương tự XGBoost, tập trung vào lớp thiểu số.
    """
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    return model
