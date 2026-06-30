import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/ml/models/fraud_detector_latest.pkl")

app = FastAPI(
    title="Fraud Detection Auth API",
    description="FastAPI service for real-time fraud prediction on login attempts.",
    version="1.0.0"
)

class SessionAttributes(BaseModel):
    transaction_amount: float
    login_attempts: int
    transaction_duration: float
    account_balance: float
    customer_age: int
    transaction_hour: int
    transaction_day_of_week: int
    transaction_month: int
    days_since_previous_transaction: float
    amount_to_balance_ratio: float
    is_large_transaction: int
    is_small_transaction: int
    is_high_risk_location: int
    is_frequent_logins: int
    is_abnormal_age: int
    is_high_risk_channel: int


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


@app.on_event("startup")
def startup_event():
    try:
        app.state.model = load_model(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model at startup: {exc}")


@app.get("/")
def health_check():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(payload: SessionAttributes):
    model = app.state.model
    features = [
        payload.transaction_amount,
        payload.login_attempts,
        payload.transaction_duration,
        payload.account_balance,
        payload.customer_age,
        payload.transaction_hour,
        payload.transaction_day_of_week,
        payload.transaction_month,
        payload.days_since_previous_transaction,
        payload.amount_to_balance_ratio,
        payload.is_large_transaction,
        payload.is_small_transaction,
        payload.is_high_risk_location,
        payload.is_frequent_logins,
        payload.is_abnormal_age,
        payload.is_high_risk_channel,
    ]

    try:
        score = model.predict([features])[0]
        probability = float(model.predict_proba([features])[0][1]) if hasattr(model, "predict_proba") else None
        return {
            "prediction": int(score),
            "probability": probability,
            "action": "block" if int(score) == 1 else "allow"
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
