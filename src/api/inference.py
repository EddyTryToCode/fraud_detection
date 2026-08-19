import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from src.feature_engineering import apply_feature_engineering
from src.explainability import get_top_influential_features
from src.llm_explainer import generate_fraud_explanation
from src.api.schemas import (
    TransactionInput, 
    PredictionResponse, 
    TopReasonItem,
    BatchPredictionResponse
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

class FraudDetectionService:
    """
    Singleton Inference Service quản lý việc nạp mô hình, tiền xử lý, 
    tính toán SHAP và sinh lời giải thích tiếng Việt.
    """
    def __init__(self, threshold: float = 0.48):
        self.threshold = threshold
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        print(f"Loading model & preprocessor from {MODELS_DIR}...")
        
        # 1. Load Preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {preprocessor_path}. Hãy chạy scripts/run_pipeline.py trước.")
        self.preprocessor = joblib.load(preprocessor_path)
        
        # 2. Load XGBoost Model
        model_path = MODELS_DIR / "xgboost_best.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {model_path}. Hãy chạy scripts/train_model.py trước.")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        
        # Lấy danh sách tên feature sau preprocessing
        feature_names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if trans != 'drop' and name != 'remainder':
                feature_names.extend(cols)
        self.feature_names = feature_names
        print("✅ FraudDetectionService initialized successfully!")

    def predict_single(self, txn: TransactionInput, use_llm: bool = True) -> PredictionResponse:
        start_time = time.time()
        
        # 1. Convert input to DataFrame
        txn_dict = txn.model_dump()
        txn_id = txn_dict.pop('transaction_id', 'UNKNOWN_TXN')
        df_raw = pd.DataFrame([txn_dict])
        
        # 2. Preprocess features
        X_df, _ = apply_feature_engineering(df_raw, is_train=False, preprocessor=self.preprocessor)
        
        # 3. Predict Probability
        risk_score = float(self.model.predict_proba(X_df)[0, 1])
        is_fraud = risk_score >= self.threshold
        decision = "FLAG_FOR_REVIEW" if is_fraud else "APPROVE"
        
        # 4. Calculate SHAP Values natively
        booster = self.model.get_booster()
        dtest = xgb.DMatrix(X_df)
        contribs = booster.predict(dtest, pred_contribs=True)
        shap_values_row = contribs[0, :-1]
        
        # 5. Extract Top Reasons
        top_reasons_raw = get_top_influential_features(
            shap_values_row=shap_values_row,
            feature_names=list(X_df.columns),
            feature_values_row=X_df.iloc[0].values,
            top_k=5
        )
        
        # 6. Generate Natural Language Explanation
        explanation_vi = generate_fraud_explanation(
            risk_score=risk_score,
            threshold=self.threshold,
            top_reasons=top_reasons_raw,
            customer_context={'age': txn.customer_age, 'income': txn.income},
            use_llm=use_llm
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Format response
        top_reasons_items = [
            TopReasonItem(
                feature=r['feature'],
                value=r['value'],
                shap_value=round(float(r['shap_value']), 4),
                impact=r['impact']
            ) for r in top_reasons_raw
        ]
        
        return PredictionResponse(
            transaction_id=txn_id,
            risk_score=round(risk_score, 4),
            threshold=self.threshold,
            decision=decision,
            is_fraud_suspected=is_fraud,
            top_reasons=top_reasons_items,
            explanation_vi=explanation_vi,
            latency_ms=round(latency_ms, 2)
        )

    def predict_batch(self, txns: List[TransactionInput], use_llm: bool = False) -> BatchPredictionResponse:
        results = [self.predict_single(t, use_llm=use_llm) for t in txns]
        flagged = sum(1 for r in results if r.is_fraud_suspected)
        approved = len(results) - flagged
        
        return BatchPredictionResponse(
            total_processed=len(results),
            flagged_count=flagged,
            approved_count=approved,
            predictions=results
        )

# Global singleton instance
service_instance: FraudDetectionService = None

def get_fraud_service() -> FraudDetectionService:
    global service_instance
    if service_instance is None:
        service_instance = FraudDetectionService()
    return service_instance
