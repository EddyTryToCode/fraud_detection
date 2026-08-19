import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame, 
    demographic_parity_difference,
    equalized_odds_difference
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import recall_score, roc_auc_score, precision_score

def compute_fairness_metrics(y_true, y_pred, y_pred_proba, sensitive_features):
    """
    Tính toán các chỉ số fairness bằng fairlearn.
    sensitive_features: pd.Series hoặc 1D array chứa thông tin nhóm (VD: độ tuổi)
    """
    
    # Định nghĩa các metrics cần tính cho từng nhóm
    metrics_dict = {
        'count': lambda y_true, y_pred: len(y_true),
        'fraud_count': lambda y_true, y_pred: sum(y_true),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'auroc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        'selection_rate': lambda y_true, y_pred: np.mean(y_pred) # Tỷ lệ bị flag là fraud
    }
    
    # Dùng MetricFrame để tính toán cho toàn bộ population và cho từng group
    mf = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Tính Disparity (Độ chênh lệch lớn nhất giữa 2 nhóm bất kỳ)
    # 1. Demographic Parity Diff: Chênh lệch tỷ lệ bị flag (selection_rate)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    
    # 2. Equal Opportunity Diff (chênh lệch Recall - True Positive Rate)
    # BAF paper tập trung vào Equal Opportunity (nhóm nào cũng phải được phát hiện fraud tốt như nhau)
    group_recalls = mf.by_group['recall']
    eo_diff = group_recalls.max() - group_recalls.min()
    
    return {
        'metric_frame': mf,
        'dp_diff': dp_diff,
        'eo_diff': eo_diff
    }

def apply_threshold_optimizer(estimator, X_train, y_train, sensitive_features_train, constraint="equalized_odds"):
    """
    Huấn luyện ThresholdOptimizer để khắc phục bias (Post-processing).
    constraint: "equalized_odds" (cân bằng Recall) hoặc "demographic_parity" (cân bằng Selection Rate).
    """
    optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraint,
        predict_method='predict_proba',
        prefit=True # Model XGBoost đã được train trước
    )
    
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    return optimizer

def compute_reweighting_weights(y, sensitive_features):
    """
    Tính toán sample weights theo phương pháp Pre-processing Reweighting (Kamiran & Calders).
    W(g, y) = (P(g) * P(y)) / P(g, y) = (N(g) * N(y)) / (N * N(g, y))
    
    Giúp triệt tiêu mối tương quan giả giữa nhóm nhạy cảm (sensitive group) và nhãn mục tiêu (target label)
    ngay trước khi đưa vào huấn luyện mô hình.
    """
    df = pd.DataFrame({'y': y, 'group': sensitive_features})
    n_total = len(df)
    
    # Tính xác suất biên
    p_group = df['group'].value_counts() / n_total
    p_y = df['y'].value_counts() / n_total
    
    # Tính xác suất đồng thời P(g, y)
    p_gy = df.groupby(['group', 'y'], observed=False).size() / n_total
    
    weights = np.zeros(n_total, dtype=float)
    for idx, (g, label) in enumerate(zip(df['group'], df['y'])):
        denom = p_gy.get((g, label), 1e-6)
        if denom > 0:
            weights[idx] = (p_group[g] * p_y[label]) / denom
        else:
            weights[idx] = 1.0
            
    # Chuẩn hóa về mean = 1.0 để giữ nguyên scale cho learning rate của XGBoost/LightGBM
    weights = weights / np.mean(weights)
    return weights

