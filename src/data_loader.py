import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def load_raw_data(variant_name="Base.csv"):
    """
    Load dữ liệu gốc từ thư mục raw.
    """
    path = RAW_DATA_DIR / variant_name
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {path}")
    
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path)
    return df

def temporal_split(df, train_months=[0,1,2,3,4,5], test_months=[6,7]):
    """
    Chia dữ liệu theo thời gian (temporal split) để giả lập production.
    - Train: Các tháng đầu (mặc định 0 -> 5)
    - Test: Các tháng sau (mặc định 6, 7)
    (Có thể tách thêm Validation set nếu cần)
    """
    print(f"Splitting data based on temporal feature (month)...")
    
    df_train = df[df['month'].isin(train_months)].copy()
    df_test = df[df['month'].isin(test_months)].copy()
    
    # Ở bài toán này, ta cũng có thể tách Month 6 làm Validation, Month 7 làm Test.
    # Nhưng tạm thời ta chia Train / Test.
    print(f"Train set: {df_train.shape[0]:,} rows (Months {train_months})")
    print(f"Test set: {df_test.shape[0]:,} rows (Months {test_months})")
    
    return df_train, df_test

def save_processed_data(df, filename):
    """
    Lưu dữ liệu đã xử lý ra định dạng Parquet (nhanh & tiết kiệm dung lượng).
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    print(f"Saving processed data to {path}...")
    df.to_parquet(path, index=False)
    
def load_processed_data(filename):
    """
    Đọc dữ liệu Parquet đã xử lý.
    """
    path = PROCESSED_DATA_DIR / filename
    return pd.read_parquet(path)
