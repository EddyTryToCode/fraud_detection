import numpy as np
import pandas as pd
import shap

def calculate_shap_values(model, X, max_samples=1000):
    """
    Tính SHAP values sử dụng XGBoost native pred_contribs.
    Khắc phục triệt để lỗi 'ValueError: could not convert string to float: [5E-1]' của SHAP.
    
    Returns:
        base_value (float): Giá trị kỳ vọng cơ sở (log-odds prior)
        shap_values (np.ndarray): Ma trận SHAP values (n_samples, n_features)
        X_sample (pd.DataFrame): DataFrame mẫu tương ứng
    """
    import xgboost as xgb
    
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X.copy()
        
    booster = model.get_booster()
    dtest = xgb.DMatrix(X_sample)
    
    # xgb predict with pred_contribs=True returns shape (n_samples, n_features + 1)
    contribs = booster.predict(dtest, pred_contribs=True)
    shap_values = contribs[:, :-1]
    base_value = float(contribs[0, -1])  # Base score ở cột cuối cùng
    
    return base_value, shap_values, X_sample

def plot_shap_summary(shap_values, X_sample, plot_type="dot"):
    """
    Vẽ SHAP Global Summary Plot (Dot plot hoặc Bar plot).
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.show()

def plot_shap_dependence(feature_name, shap_values, X_sample, interaction_index="auto"):
    """
    Vẽ SHAP Dependence Plot để xem sự tương tác phi tuyến giữa các đặc trưng.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature_name, shap_values, X_sample, interaction_index=interaction_index, show=False)
    plt.tight_layout()
    plt.show()

def plot_shap_waterfall(shap_values, X_sample, row_idx=0, base_value=0.0, max_display=10, title=None):
    """
    Vẽ SHAP Local Explanation Waterfall Plot cho 1 giao dịch cụ thể.
    """
    import matplotlib.pyplot as plt
    feature_names = list(X_sample.columns) if hasattr(X_sample, 'columns') else None
    data_row = X_sample.iloc[row_idx].values if hasattr(X_sample, 'iloc') else X_sample[row_idx]
    
    exp = shap.Explanation(
        values=shap_values[row_idx],
        base_values=base_value,
        data=data_row,
        feature_names=feature_names
    )
    
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12, pad=15)
    plt.tight_layout()
    plt.show()

def get_top_influential_features(shap_values_row, feature_names, feature_values_row, top_k=5):
    """
    Trích xuất top K đặc trưng tác động mạnh nhất đến quyết định của một giao dịch đơn lẻ.
    Trả về cấu trúc dữ liệu rõ ràng để phục vụ hiển thị Dashboard và làm đầu vào cho LLM.
    """
    records = []
    for f_name, f_val, s_val in zip(feature_names, feature_values_row, shap_values_row):
        records.append({
            'feature': f_name,
            'value': f_val,
            'shap_value': s_val,
            'abs_shap': abs(s_val),
            'impact': 'TĂNG RỦI RO (Fraud)' if s_val > 0 else 'GIẢM RỦI RO (Legitimate)'
        })
    
    df = pd.DataFrame(records).sort_values(by='abs_shap', ascending=False)
    return df.head(top_k).to_dict(orient='records')
