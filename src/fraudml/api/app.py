import os
import sys
import time
import uuid
import json
import joblib
import hashlib
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, List

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..data.features import engineer_features, get_feature_names
from ..monitoring.metrics import (
    record_prediction, record_api_request, get_metrics, get_content_type
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

# Thread pool for CPU-bound model inference
executor = ThreadPoolExecutor(max_workers=4)

model = None
meta = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, meta
    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    meta_path = os.getenv("META_PATH", "artifacts/model_meta.json")

    if not os.path.exists(model_path):
        logger.info(f"Artifacts not found at {model_path}. Bootstrapping model...")
        try:
            logger.info("Running data download...")
            subprocess.check_call([sys.executable, "src/fraudml/data/download.py"])

            logger.info("Running preprocessing...")
            subprocess.check_call([sys.executable, "src/fraudml/data/preprocess.py"])

            logger.info("Running training...")
            subprocess.check_call([sys.executable, "src/fraudml/models/train.py"])

            logger.info("Bootstrap complete. Model trained.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during bootstrap: {e}")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"CRITICAL: Model not found at {model_path} even after bootstrap attempt.")

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        logger.info(f"Loaded metadata from {meta_path}")

    yield

app = FastAPI(
    title="Fraud Detection API v2",
    description="Production-ready fraud detection with ensemble models, SHAP explanations, and monitoring",
    version="2.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter

# Add GZip compression for responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: str = Header(...)):
    valid_key = os.getenv("API_KEY", "default-api-key")
    if x_api_key != valid_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

def get_client_ip(request: Request) -> str:
    return get_remote_address(request)


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    device_score: float = Field(..., ge=0, le=1, description="Device reliability score")
    country_risk: int = Field(..., ge=1, le=5, description="Country risk level")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    transaction_count_last_hour: Optional[int] = Field(0, ge=0, description="Recent transaction count")


class BatchTransaction(BaseModel):
    transactions: list[Transaction]


class ExplanationResponse(BaseModel):
    fraud_probability: float
    fraud_label: int
    model_version: str
    model_type: str
    threshold: float
    prediction_id: str
    processing_time_ms: float
    timestamp: str
    explanation: dict


class BatchResponse(BaseModel):
    predictions: list[dict]
    batch_id: str
    total_processing_time_ms: float
    model_version: str
    model_type: str
    timestamp: str


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": meta.get("model_version") if meta else None,
        "model_type": meta.get("model_type") if meta else None,
        "api_version": "2.0.0"
    }


@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    return Response(content=get_metrics(), media_type=get_content_type())


@app.post("/v1/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def predict(request: Request, transaction: Transaction):
    start_time = time.time()
    prediction_id = str(uuid.uuid4())

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feature_cols = get_feature_names()
        data = transaction.model_dump()
        df = pd.DataFrame([data])
        engineered = engineer_features(df)
        features = engineered[feature_cols]

        prob = model.predict_proba(features)[0, 1]
        threshold = meta.get("threshold", 0.5) if meta else 0.5
        label = int(prob >= threshold)

        processing_time = (time.time() - start_time) * 1000

        result = {
            "fraud_probability": float(prob),
            "fraud_label": label,
            "model_version": meta.get("model_version") if meta else "unknown",
            "model_type": meta.get("model_type") if meta else "unknown",
            "threshold": threshold,
            "prediction_id": prediction_id,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        record_prediction(
            model_type=meta.get("model_type", "unknown"),
            fraud_label=label,
            latency=processing_time / 1000
        )

        logger.info(f"Prediction {prediction_id}: fraud={label}, prob={prob:.4f}")

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/batch-predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("50/minute")
async def batch_predict(request: Request, batch: BatchTransaction):
    """Batch prediction with vectorized processing for better throughput."""
    start_time = time.time()
    batch_id = str(uuid.uuid4())

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feature_cols = get_feature_names()
        threshold = meta.get("threshold", 0.5) if meta else 0.5

        def process_batch():
            """Vectorized batch processing - runs in thread pool."""
            # Build DataFrame from all transactions at once
            batch_data = [tx.model_dump() for tx in batch.transactions]
            df = pd.DataFrame(batch_data)
            
            # Engineer features for entire batch
            engineered = engineer_features(df)
            features = engineered[feature_cols]
            
            # Vectorized prediction for all samples
            probs = model.predict_proba(features)[:, 1]
            labels = (probs >= threshold).astype(int)
            
            return [
                {"fraud_probability": float(p), "fraud_label": int(l)}
                for p, l in zip(probs, labels)
            ]

        # Run CPU-bound prediction in thread pool
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(executor, process_batch)

        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch {batch_id}: processed {len(predictions)} in {total_time:.2f}ms")

        return {
            "predictions": predictions,
            "batch_id": batch_id,
            "total_processing_time_ms": round(total_time, 2),
            "model_version": meta.get("model_version") if meta else "unknown",
            "model_type": meta.get("model_type") if meta else "unknown",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/explain", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def explain(request: Request, transaction: Transaction):
    start_time = time.time()
    prediction_id = str(uuid.uuid4())

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import shap

        feature_cols = get_feature_names()
        data = transaction.model_dump()
        df = pd.DataFrame([data])
        engineered = engineer_features(df)
        features = engineered[feature_cols]

        prob = model.predict_proba(features)[0, 1]
        threshold = meta.get("threshold", 0.5) if meta else 0.5
        label = int(prob >= threshold)

        if hasattr(model, 'predict_proba'):
            if hasattr(shap, 'KernelExplainer'):
                try:
                    explainer = shap.KernelExplainer(model.predict_proba, features)
                    shap_values = explainer.shap_values(features)

                    base_value = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, list) else float(explainer.expected_value)

                    feature_descriptions = {
                        "amount": "High transaction amount increases fraud risk",
                        "hour": "Late night transactions are higher risk",
                        "device_score": "Low device trust score increases risk",
                        "country_risk": "High-risk country contributes to fraud probability",
                        "amount_log": "Log-transformed amount reduces outlier impact",
                        "is_night": "Nighttime transactions are flagged",
                        "high_amount": "Amount exceeds 75th percentile",
                        "low_device": "Device reliability is below 0.3"
                    }

                    contributions = []
                    for i, feat in enumerate(feature_cols):
                        if i < len(shap_values[0]):
                            contrib = float(shap_values[0][i]) if isinstance(shap_values[0], (list, np.ndarray)) else 0.0
                            contributions.append({
                                "feature": feat,
                                "value": float(data.get(feat, engineered[feat].values[0])),
                                "contribution": round(contrib, 4),
                                "description": feature_descriptions.get(feat, "")
                            })

                    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

                    processing_time = (time.time() - start_time) * 1000

                    return {
                        "fraud_probability": float(prob),
                        "fraud_label": label,
                        "model_version": meta.get("model_version") if meta else "unknown",
                        "model_type": meta.get("model_type") if meta else "unknown",
                        "threshold": threshold,
                        "prediction_id": prediction_id,
                        "processing_time_ms": round(processing_time, 2),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "explanation": {
                            "base_value": base_value,
                            "feature_contributions": contributions[:10]
                        }
                    }
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            else:
                logger.warning("SHAP library not available")

        processing_time = (time.time() - start_time) * 1000

        return {
            "fraud_probability": float(prob),
            "fraud_label": label,
            "model_version": meta.get("model_version") if meta else "unknown",
            "model_type": meta.get("model_type") if meta else "unknown",
            "threshold": threshold,
            "prediction_id": prediction_id,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "explanation": {
                "base_value": 0.05,
                "feature_contributions": [],
                "message": "SHAP explanations not available for this model"
            }
        }

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
