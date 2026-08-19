from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TransactionInput(BaseModel):
    # Định danh giao dịch (tùy chọn)
    transaction_id: Optional[str] = Field(default="TXN_DEMO_001", description="Mã định danh giao dịch")
    
    # Numerical Features (19 features)
    income: float = Field(default=0.5, ge=0.0, le=1.0, description="Mức thu nhập chuẩn hóa (0.1 - 0.9)")
    name_email_similarity: float = Field(default=0.45, ge=0.0, le=1.0, description="Độ tương đồng tên và email")
    prev_address_months_count: float = Field(default=-1.0, description="Số tháng ở địa chỉ trước (-1 nếu không có)")
    current_address_months_count: float = Field(default=36.0, ge=-1.0, description="Số tháng ở địa chỉ hiện tại")
    customer_age: float = Field(default=35.0, ge=10.0, le=100.0, description="Độ tuổi khách hàng (10 - 90)")
    days_since_request: float = Field(default=0.01, ge=0.0, description="Số ngày kể từ lúc gửi yêu cầu")
    intended_balcon_amount: float = Field(default=15.0, description="Số dư chuyển khoản dự kiến")
    zip_count_4w: float = Field(default=1200.0, ge=0.0, description="Số đơn cùng zip code trong 4 tuần")
    velocity_6h: float = Field(default=2500.0, ge=0.0, description="Tần suất thao tác trong 6 giờ")
    velocity_24h: float = Field(default=4000.0, ge=0.0, description="Tần suất thao tác trong 24 giờ")
    velocity_4w: float = Field(default=5000.0, ge=0.0, description="Tần suất thao tác trong 4 tuần")
    bank_branch_count_8w: float = Field(default=0.0, ge=0.0, description="Số chi nhánh giao dịch trong 8 tuần")
    date_of_birth_distinct_emails_4w: float = Field(default=5.0, ge=0.0, description="Số email khác nhau cùng ngày sinh")
    credit_risk_score: float = Field(default=110.0, description="Điểm rủi ro tín dụng nội bộ")
    bank_months_count: float = Field(default=12.0, ge=-1.0, description="Thời gian quan hệ ngân hàng (tháng)")
    proposed_credit_limit: float = Field(default=500.0, ge=0.0, description="Hạn mức tín dụng đề xuất")
    session_length_in_minutes: float = Field(default=8.5, ge=0.0, description="Thời lượng phiên (phút)")
    device_distinct_emails_8w: float = Field(default=1.0, ge=0.0, description="Số email liên kết thiết bị trong 8 tuần")
    device_fraud_count: float = Field(default=0.0, ge=0.0, description="Số lần thiết bị bị đánh dấu gian lận")
    
    # Categorical Features (5 features)
    payment_type: str = Field(default="AB", description="Hình thức thanh toán (AA, AB, AC, AD, AE)")
    employment_status: str = Field(default="CA", description="Tình trạng việc làm (CA, CB, CC, CD, CE, CF, CG)")
    housing_status: str = Field(default="BC", description="Tình trạng nhà ở (BA, BB, BC, BD, BE, BF, BG)")
    source: str = Field(default="INTERNET", description="Kênh nộp hồ sơ (INTERNET, TELEAPP)")
    device_os: str = Field(default="windows", description="Hệ điều hành thiết bị (windows, macintosh, linux, x11, other)")
    
    # Binary Features (6 features)
    email_is_free: int = Field(default=1, ge=0, le=1, description="1 nếu dùng email miễn phí, 0 nếu không")
    phone_home_valid: int = Field(default=1, ge=0, le=1, description="Tính hợp lệ số điện thoại bàn")
    phone_mobile_valid: int = Field(default=1, ge=0, le=1, description="Tính hợp lệ số điện thoại di động")
    has_other_cards: int = Field(default=0, ge=0, le=1, description="1 nếu đã có thẻ ngân hàng khác")
    foreign_request: int = Field(default=0, ge=0, le=1, description="1 nếu request từ nước ngoài / VPN")
    keep_alive_session: int = Field(default=0, ge=0, le=1, description="Tín hiệu duy trì phiên")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_SUSPICIOUS_999",
                "income": 0.1,
                "name_email_similarity": 0.05,
                "prev_address_months_count": -1.0,
                "current_address_months_count": 2.0,
                "customer_age": 70.0,
                "days_since_request": 0.002,
                "intended_balcon_amount": 50.0,
                "zip_count_4w": 3500.0,
                "velocity_6h": 8500.0,
                "velocity_24h": 9200.0,
                "velocity_4w": 9800.0,
                "bank_branch_count_8w": 5.0,
                "date_of_birth_distinct_emails_4w": 25.0,
                "credit_risk_score": 280.0,
                "bank_months_count": -1.0,
                "proposed_credit_limit": 1500.0,
                "session_length_in_minutes": 0.5,
                "device_distinct_emails_8w": 6.0,
                "device_fraud_count": 1.0,
                "payment_type": "AC",
                "employment_status": "CF",
                "housing_status": "BE",
                "source": "INTERNET",
                "device_os": "linux",
                "email_is_free": 1,
                "phone_home_valid": 0,
                "phone_mobile_valid": 1,
                "has_other_cards": 0,
                "foreign_request": 1,
                "keep_alive_session": 0
            }
        }


class TopReasonItem(BaseModel):
    feature: str
    value: Any
    shap_value: float
    impact: str


class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float
    threshold: float
    decision: str  # "APPROVE" hoặc "FLAG_FOR_REVIEW"
    is_fraud_suspected: bool
    top_reasons: List[TopReasonItem]
    explanation_vi: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    flagged_count: int
    approved_count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    sota_benchmark_auroc: float
    fairness_eod_debiased: float
    features_count: int
