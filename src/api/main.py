from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from src.api.inference import get_fraud_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động và làm nóng (warm up) model
    print("🚀 Khởi động Banking Fraud Detection API...")
    service = get_fraud_service()
    print(f"✅ Hệ thống đã sẵn sàng phục vụ tại ngưỡng Threshold = {service.threshold}!")
    yield
    print("🛑 Đang đóng API Service...")

app = FastAPI(
    title="🏦 Bank Account Fraud (BAF) Detection API",
    description="""
    Hệ thống phát hiện gian lận mở tài khoản ngân hàng đạt chuẩn **NeurIPS 2022 Benchmark** (XGBoost AUROC 0.8895).
    
    ### Tính năng chính:
    - ⚡ **Real-time Scoring**: Dự đoán xác suất gian lận siêu tốc (< 25ms).
    - ⚖️ **Fairness & Cost Optimization**: Tối ưu ngưỡng theo chi phí vận hành (1:50) và cân bằng thiên vị (Fairlearn).
    - 🔍 **SHAP Local Explanation**: Bóc tách top 5 yếu tố ảnh hưởng trực tiếp đến quyết định.
    - 🤖 **LLM Translation Layer**: Tự động chuyển đổi SHAP kỹ thuật thành báo cáo nghiệp vụ tiếng Việt cho Compliance Team.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Bank Account Fraud Detection API",
        "status": "online",
        "documentation": "/docs",
        "sota_auroc": 0.8895,
        "optimal_threshold": 0.48
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    service = get_fraud_service()
    return HealthResponse(
        status="HEALTHY",
        model_name="XGBoost Classifier (NeurIPS 2022 SOTA)",
        model_version="1.0.0",
        sota_benchmark_auroc=0.8895,
        fairness_eod_debiased=0.0000,
        features_count=30
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_transaction(
    transaction: TransactionInput,
    use_llm: bool = Query(default=True, description="Bật sinh giải thích tiếng Việt tự nhiên qua LLM/NLG Layer")
):
    """
    Chấm điểm rủi ro gian lận cho 1 giao dịch đơn lẻ và sinh báo cáo giải trình tiếng Việt.
    """
    try:
        service = get_fraud_service()
        response = service.predict_single(transaction, use_llm=use_llm)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý dự đoán: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_transactions(
    request: BatchPredictionRequest,
    use_llm: bool = Query(default=False, description="Bật sinh giải thích cho từng dòng trong batch")
):
    """
    Xử lý hàng loạt (Batch Processing) cho nhiều giao dịch cùng lúc.
    """
    try:
        service = get_fraud_service()
        response = service.predict_batch(request.transactions, use_llm=use_llm)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý batch: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
