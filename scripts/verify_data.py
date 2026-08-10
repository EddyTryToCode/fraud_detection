"""
Kiểm tra nhanh dữ liệu BAF đã tải đúng và hiểu đúng cấu trúc.

Chạy: python scripts/verify_data.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# BAF có 6 variant: Base, Variant I - V (mỗi variant có 1 loại bias khác nhau)
DEFAULT_VARIANT = "Base.csv"


def main():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ Không tìm thấy file .csv nào trong {RAW_DATA_DIR}")
        print("   → Chạy trước: python scripts/download_data.py")
        return

    print(f"📂 Tìm thấy {len(csv_files)} file:")
    for f in csv_files:
        print(f"   - {f.name}")

    target = RAW_DATA_DIR / DEFAULT_VARIANT
    if not target.exists():
        target = csv_files[0]
        print(f"\n⚠️  Không thấy {DEFAULT_VARIANT}, dùng thử {target.name} thay thế")

    print(f"\n🔍 Đang load {target.name} ...")
    df = pd.read_csv(target)

    print(f"\n--- Tổng quan dataset: {target.name} ---")
    print(f"Shape: {df.shape[0]:,} dòng x {df.shape[1]} cột")

    if "fraud_bool" in df.columns:
        fraud_count = df["fraud_bool"].sum()
        fraud_rate = fraud_count / len(df) * 100
        print(f"\nSố case fraud: {fraud_count:,} / {len(df):,} ({fraud_rate:.2f}%)")
        print(f"→ Đây là dữ liệu mất cân bằng nghiêm trọng (extreme imbalance),")
        print(f"  đúng như mô tả trong paper gốc — cần xử lý cẩn thận ở bước modeling.")
    else:
        print("\n⚠️  Không thấy cột 'fraud_bool' — kiểm tra lại tên cột nhãn.")

    print(f"\n--- Danh sách cột ---")
    for col in df.columns:
        print(f"   - {col} ({df[col].dtype})")

    print(f"\n--- 3 dòng đầu ---")
    print(df.head(3).to_string())

    print("\n✅ Dữ liệu load thành công. Sẵn sàng cho bước feature engineering.")


if __name__ == "__main__":
    main()
