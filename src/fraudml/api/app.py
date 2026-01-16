import os
import joblib
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Global variables for model artifacts
model = None
meta = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and metadata on startup
    global model, meta
    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    meta_path = os.getenv("META_PATH", "artifacts/model_meta.json")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"Loaded metadata from {meta_path}")
    
    yield
    
    # Clean up (if needed)

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

class Transaction(BaseModel):
    amount: float
    hour: int
    device_score: float
    country_risk: int

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": meta.get("model_version") if meta else None
    }

@app.post("/predict")
def predict(transaction: Transaction):
    global model, meta
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Create DataFrame for prediction to match training columns
    # Ensure correct order: amount, hour, device_score, country_risk
    # (Checking meta['features'] would be even better for safety)
    features = pd.DataFrame([{
        "amount": transaction.amount,
        "hour": transaction.hour,
        "device_score": transaction.device_score,
        "country_risk": transaction.country_risk
    }])
    
    # Predict probability
    prob = model.predict_proba(features)[0, 1]
    
    # Predict label based on tuned threshold (from meta)
    # If not found, fallback to 0.5 (though train.py now guarantees it's in meta)
    threshold = meta.get("threshold", 0.5) if meta else 0.5
    label = int(prob >= threshold)
    
    return {
        "fraud_probability": float(prob),
        "fraud_label": label,
        "model_version": meta.get("model_version") if meta else "unknown",
        "threshold": threshold
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
