"""
Tải bộ dữ liệu Bank Account Fraud (BAF) từ Kaggle.

Yêu cầu trước khi chạy:
1. Đã cài `kaggle` package (có trong requirements.txt)
2. Đã đặt file kaggle.json vào ~/.kaggle/kaggle.json (Linux/Mac)
   hoặc C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)
3. Đã chấp nhận điều khoản sử dụng dataset trên trang Kaggle
   (bắt buộc — vào link dataset, bấm "Download" 1 lần trên web
   để accept rules trước khi dùng API)

Chạy: python scripts/download_data.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_SLUG = "sgpjesus/bank-account-fraud-dataset-neurips-2022"


def check_kaggle_credentials():
    """Kiểm tra xem đã cấu hình Kaggle API credentials chưa."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Không tìm thấy ~/.kaggle/kaggle.json")
        print("   → Vào https://www.kaggle.com/settings → Create New Token")
        print("   → Tải file kaggle.json, đặt vào ~/.kaggle/kaggle.json")
        print("   → Chạy: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("✅ Tìm thấy Kaggle credentials")


def download_dataset():
    """Tải dataset BAF về data/raw/ bằng Kaggle CLI."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Đang tải dataset '{DATASET_SLUG}' về {RAW_DATA_DIR} ...")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(RAW_DATA_DIR),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Tải thất bại. Chi tiết lỗi:")
        print(result.stderr)
        print("\nGợi ý: nếu lỗi 403, bạn cần vào trang dataset trên Kaggle")
        print(f"https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print("và bấm nút Download 1 lần trên web để accept rules trước.")
        sys.exit(1)

    print("✅ Tải xong. Danh sách file:")
    for f in sorted(RAW_DATA_DIR.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    check_kaggle_credentials()
    download_dataset()
