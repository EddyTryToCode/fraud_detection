from xgboost import XGBClassifier

def build_xgboost_model(scale_pos_weight=1.0):
    """
    Tạo mô hình XGBoost cho Fraud Detection.
    - scale_pos_weight: Giúp mô hình tập trung vào lớp thiểu số (fraud).
    - eval_metric: 'aucpr' (Area Under the PR Curve) rất quan trọng cho imbalanced data.
    """
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    return model
