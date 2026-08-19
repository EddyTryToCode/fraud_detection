from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_logistic_baseline():
    """
    Tạo mô hình Logistic Regression làm Baseline.
    - class_weight='balanced': Tự động xử lý imbalance bằng cách
      tăng trọng số cho lớp thiểu số (fraud).
    - max_iter: Tăng lên để đảm bảo hội tụ.
    """
    model = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000,
        random_state=42
    )
    return model
