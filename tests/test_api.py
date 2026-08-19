import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.main import app
from src.api.schemas import TransactionInput

print("Testing FastAPI app locally with TestClient...")
client = TestClient(app)

# 1. Test Health
print("\n1. Testing GET /health...")
res = client.get("/health")
print("Status Code:", res.status_code)
print("Response:", res.json())
assert res.status_code == 200

# 2. Test Predict Suspicious Case
print("\n2. Testing POST /predict (Suspicious Case)...")
suspicious_payload = {
    "transaction_id": "TXN_TEST_FRAUD_001",
    "income": 0.1,
    "name_email_similarity": 0.02,
    "current_address_months_count": 1.0,
    "customer_age": 75.0,
    "device_distinct_emails_8w": 8.0,
    "device_fraud_count": 2.0,
    "email_is_free": 1,
    "foreign_request": 1,
    "credit_risk_score": 320.0
}
res_pred = client.post("/predict", json=suspicious_payload)
print("Status Code:", res_pred.status_code)
pred_json = res_pred.json()
print(f"Risk Score: {pred_json['risk_score']:.2%}")
print(f"Decision: {pred_json['decision']}")
print(f"Latency: {pred_json['latency_ms']} ms")
print(f"Top Reasons Count: {len(pred_json['top_reasons'])}")
print("\nExplanation:\n", pred_json['explanation_vi'])
assert res_pred.status_code == 200

# 3. Test Predict Normal Case
print("\n3. Testing POST /predict (Normal Safe Case)...")
normal_payload = {
    "transaction_id": "TXN_TEST_SAFE_002",
    "income": 0.8,
    "name_email_similarity": 0.95,
    "current_address_months_count": 120.0,
    "customer_age": 32.0,
    "device_distinct_emails_8w": 1.0,
    "device_fraud_count": 0.0,
    "email_is_free": 0,
    "foreign_request": 0,
    "credit_risk_score": 85.0
}
res_normal = client.post("/predict", json=normal_payload)
print("Status Code:", res_normal.status_code)
normal_json = res_normal.json()
print(f"Risk Score: {normal_json['risk_score']:.2%}")
print(f"Decision: {normal_json['decision']}")
print(f"Latency: {normal_json['latency_ms']} ms")
print("\nExplanation:\n", normal_json['explanation_vi'])
assert res_normal.status_code == 200

print("\n🎉 ALL API TESTS PASSED PERFECTLY!")
