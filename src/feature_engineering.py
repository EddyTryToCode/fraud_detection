import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

# Các feature được định nghĩa dựa trên EDA
NUM_COLS = [
    'income', 'name_email_similarity', 'prev_address_months_count', 
    'current_address_months_count', 'customer_age', 'days_since_request', 
    'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 
    'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 
    'credit_risk_score', 'bank_months_count', 'proposed_credit_limit', 
    'session_length_in_minutes', 'device_distinct_emails_8w', 'device_fraud_count'
]

CAT_COLS = [
    'payment_type', 'employment_status', 'housing_status', 'source', 'device_os'
]

BIN_COLS = [
    'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 
    'has_other_cards', 'foreign_request', 'keep_alive_session'
]

SENTINEL_COLS = [
    'prev_address_months_count', 
    'current_address_months_count', 
    'bank_months_count'
]


class MissingValueIndicator(BaseEstimator, TransformerMixin):
    """
    Tạo binary features (indicators) cho các giá trị sentinel (-1) 
    và thay thế -1 bằng NaN để xử lý chuẩn xác hơn trong scaling.
    """
    def __init__(self, sentinel_value=-1, cols=None):
        self.sentinel_value = sentinel_value
        self.cols = cols if cols else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            if col in X_out.columns:
                # Tạo cột indicator
                indicator_col = f"{col}_is_missing"
                X_out[indicator_col] = (X_out[col] == self.sentinel_value).astype(int)
                
                # Replace -1 with NaN để imputer hoặc model tự xử lý tốt hơn thay vì coi -1 là số nguyên
                # Trong bài toán này, ta có thể giữ nguyên NaN vì XGBoost/LightGBM hỗ trợ NaN native.
                X_out[col] = X_out[col].replace(self.sentinel_value, np.nan)
        return X_out


def get_feature_pipeline():
    """
    Tạo sklearn Pipeline cho Feature Engineering.
    """
    # Xử lý Numerical features: Impute NaN (từ sentinel) bằng median và Scale
    from sklearn.impute import SimpleImputer
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Xử lý Categorical features: Ordinal Encoding (tốt cho Tree-based models)
    cat_pipeline = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_COLS),
        ('cat', cat_pipeline, CAT_COLS),
        ('bin', 'passthrough', BIN_COLS) 
        # Các cột indicator tạo từ MissingValueIndicator sẽ tự động giữ lại
        # nếu ta áp dụng MissingValueIndicator ở bước trước ColumnTransformer
    ], remainder='passthrough')

    return preprocessor

def apply_feature_engineering(df, is_train=True, preprocessor=None):
    """
    Hàm wrapper để áp dụng toàn bộ logic FE lên DataFrame.
    """
    # 1. Tạo Missing Indicators
    missing_indicator = MissingValueIndicator(cols=SENTINEL_COLS)
    df_processed = missing_indicator.transform(df)
    
    # Danh sách các cột indicator vừa tạo
    indicator_cols = [f"{c}_is_missing" for c in SENTINEL_COLS if c in df.columns]
    
    # Tách X, y
    y = df_processed['fraud_bool'] if 'fraud_bool' in df_processed.columns else None
    
    # Lấy các cột feature, bỏ target và bỏ cột 'month' (nếu không muốn dùng 'month' làm feature)
    # Tuy nhiên 'month' có thể dùng để model học được trend, 
    # nhưng để tránh overfitting theo tháng, thường ta bỏ 'month' ra khỏi training.
    drop_cols = ['fraud_bool', 'month']
    X = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])
    
    # Cập nhật thứ tự các cột vào preprocessor
    if is_train:
        preprocessor = get_feature_pipeline()
        # Thay đổi ColumnTransformer một chút để pass through indicator cols
        preprocessor.transformers.append(('indicators', 'passthrough', indicator_cols))
        
        X_transformed = preprocessor.fit_transform(X)
    else:
        if preprocessor is None:
            raise ValueError("Cần truyền preprocessor (đã fit) khi is_train=False")
        X_transformed = preprocessor.transform(X)
        
    # Chuyển lại về DataFrame để dễ nhìn/lưu dạng Parquet
    # Lấy danh sách feature names từ preprocessor
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if trans != 'drop' and name != 'remainder':
            feature_names.extend(cols)
    
    # remainder cols (nếu có cột nào chưa đc chỉ định)
    remainder_cols = [c for c in X.columns if c not in feature_names]
    all_feature_names = feature_names + remainder_cols
    
    X_df = pd.DataFrame(X_transformed, columns=all_feature_names, index=X.index)
    
    # Gắn lại y và month (nếu cần tracking)
    if y is not None:
        X_df['fraud_bool'] = y
    if 'month' in df_processed.columns:
        X_df['month'] = df_processed['month']
        
    return X_df, preprocessor
