import numpy as np

def calculate_scale_pos_weight(y_train):
    """
    Tính toán scale_pos_weight cho XGBoost/LightGBM.
    Công thức: sum(negative instances) / sum(positive instances)
    """
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    
    if n_positive == 0:
        return 1.0
        
    weight = n_negative / n_positive
    return weight

def get_class_weights(y_train):
    """
    Tính class_weight dictionary cho Logistic Regression hoặc sklearn models.
    """
    n_samples = len(y_train)
    n_classes = 2
    
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    
    weight_0 = n_samples / (n_classes * n_negative)
    weight_1 = n_samples / (n_classes * n_positive)
    
    return {0: weight_0, 1: weight_1}
