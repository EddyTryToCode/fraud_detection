import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score, confusion_matrix
)

def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Đánh giá mô hình phân loại nhị phân (Fraud Detection).
    Trả về Dictionary chứa các metrics quan trọng.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 1. Các metric phụ thuộc ngưỡng (Threshold-dependent)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 3. Các metric không phụ thuộc ngưỡng (Threshold-independent)
    auroc = roc_auc_score(y_true, y_pred_proba)
    # Tương đương PR-AUC (Area Under Precision-Recall Curve)
    pr_auc = average_precision_score(y_true, y_pred_proba) 
    
    metrics = {
        'AUROC': auroc,
        'PR-AUC': pr_auc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'False Positives': fp,
        'False Negatives': fn
    }
    return metrics

def compare_models(results_dict):
    """
    So sánh kết quả các model dạng DataFrame.
    """
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    # Sắp xếp theo AUROC giảm dần
    return df.sort_values(by='AUROC', ascending=False)
