import os
import sys
import time
import uuid
import json
import joblib
import hashlib
import asyncio
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, List

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .cache import (
    prediction_cache, model_cache, request_queue,
    get_cached_or_predict, AsyncPredictionCache
)
from .auth import (
    auth_handler, audit_logger, get_current_user, TokenPayload,
    verify_api_authentication, api_key_auth, require_permission
)
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
            subprocess.check_call([sys.executable, "src/fraudml/models/train.py", "--model", "XGBoost"])

            logger.info("Bootstrap complete. Model trained.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during bootstrap: {e}")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            logger.info(f"Loaded metadata from {meta_path}")
        
        try:
            await prediction_cache.warm_cache([
                {"amount": 100, "hour": 12, "device_score": 0.9, "country_risk": 1},
                {"amount": 500, "hour": 14, "device_score": 0.7, "country_risk": 2},
                {"amount": 1000, "hour": 18, "device_score": 0.5, "country_risk": 3},
            ], model, meta or {})
        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")
    else:
        logger.error(f"CRITICAL: Model not found at {model_path}")

    yield

    await prediction_cache.clear_all()

app = FastAPI(
    title="Fraud Detection API v3",
    description="Production-ready fraud detection with async processing, JWT auth, and caching",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.state.limiter = limiter

# Add GZip compression for responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def sanitize_input(data: Dict) -> Dict:
    """Sanitize input data to prevent injection attacks."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            value = value.strip()[:1000]
        sanitized[key] = value
    return sanitized


class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, le=1000000, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    device_score: float = Field(..., ge=0, le=1, description="Device reliability score")
    country_risk: int = Field(..., ge=1, le=5, description="Country risk level (1-5)")
    merchant_id: Optional[str] = Field(None, max_length=100, description="Merchant identifier")
    user_id: Optional[str] = Field(None, max_length=100, description="User identifier")
    transaction_count_last_hour: Optional[int] = Field(0, ge=0, le=1000, description="Recent transaction count")
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]
    batch_priority: Optional[str] = Field("normal", description="Processing priority")


class ExplanationRequest(BaseModel):
    include_details: bool = Field(True, description="Include detailed explanations")


class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_label: int
    model_version: str
    model_type: str
    threshold: float
    prediction_id: str
    processing_time_ms: float
    timestamp: str
    cache_hit: bool = False


class BatchResponse(BaseModel):
    predictions: List[Dict]
    batch_id: str
    total_processing_time_ms: float
    model_version: str
    model_type: str
    timestamp: str
    cached_count: int = 0


class ExplanationResponse(BaseModel):
    fraud_probability: float
    fraud_label: int
    model_version: str
    model_type: str
    threshold: float
    prediction_id: str
    processing_time_ms: float
    timestamp: str
    explanation: Dict


class CacheStatsResponse(BaseModel):
    total_keys: int
    hits: int
    misses: int
    hit_rate: float


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


async def check_rate_limit(user_id: str, endpoint: str) -> bool:
    """Check per-user rate limit."""
    max_requests = 100
    window = 60
    
    request_queue.record_request(user_id)
    request_count = request_queue.get_request_count(user_id, window)
    
    if request_count > max_requests:
        audit_logger.log_auth_failure(
            "rate_limit_exceeded",
            {"endpoint": endpoint, "request_count": request_count},
            get_client_ip(Request)
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_requests} requests per minute."
        )
    return True


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": meta.get("model_version") if meta else None,
        "model_type": meta.get("model_type") if meta else None,
        "api_version": "3.0.0"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    return Response(content=get_metrics(), media_type=get_content_type())


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics."""
    stats = await prediction_cache.get_stats()
    return CacheStatsResponse(**stats)


@app.post("/v1/auth/token", response_model=AuthTokenResponse)
async def get_access_token(user_id: str = Header(...), api_key: str = Header(...)):
    """Get access token using API key."""
    user_data = auth_handler.verify_api_key(api_key)
    if not user_data:
        audit_logger.log_auth_failure("invalid_api_key", {"user_id": user_id})
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    access_token, expires = auth_handler.create_access_token(
        user_id,
        user_data.get("permissions", [])
    )
    refresh_token, _ = auth_handler.create_refresh_token(user_id)
    
    audit_logger.log("token_generated", user_id, {"permissions": user_data.get("permissions", [])})
    
    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires - int(time.time())
    )


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    transaction: TransactionInput,
    credentials: HTTPAuthorizationCredentials = Security(security),
    api_key: str = Header(None)
):
    """Single fraud prediction with JWT auth and caching."""
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        auth_result = await verify_api_authentication(api_key, credentials)
        user_id = auth_result["data"].user_id if hasattr(auth_result["data"], "user_id") else "api_user"
        
        await check_rate_limit(user_id, "/v1/predict")
        
        features = sanitize_input(transaction.model_dump())
        
        result = await get_cached_or_predict(model, meta or {}, features)
        result["prediction_id"] = prediction_id
        result["cache_hit"] = result.get("cache_hit", False)
        
        processing_time = (time.time() - start_time) * 1000
        
        audit_logger.log_prediction(
            user_id=user_id,
            transaction=features,
            result=result,
            model_version=meta.get("model_version") if meta else "unknown",
            latency_ms=processing_time,
            ip_address=get_client_ip(request)
        )
        
        record_prediction(
            model_type=meta.get("model_type", "unknown") if meta else "unknown",
            fraud_label=result["fraud_label"],
            latency=processing_time / 1000
        )
        
        logger.info(f"Prediction {prediction_id}: fraud={result['fraud_label']}, prob={result['fraud_probability']:.4f}")
        
        return PredictionResponse(
            **result,
            processing_time_ms=round(processing_time, 2),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        
    except HTTPException:
        raise
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
        auth_result = await verify_api_authentication(api_key, credentials)
        user_id = auth_result["data"].user_id if hasattr(auth_result["data"], "user_id") else "api_user"
        
        await check_rate_limit(user_id, "/v1/batch-predict")
        
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


@app.post("/v1/explain", response_model=ExplanationResponse)
async def explain(
    request: Request,
    transaction: TransactionInput,
    credentials: HTTPAuthorizationCredentials = Security(security),
    api_key: str = Header(None)
):
    """Prediction with SHAP explainability."""
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        auth_result = await verify_api_authentication(api_key, credentials)
        user_id = auth_result["data"].user_id if hasattr(auth_result["data"], "user_id") else "api_user"
        
        await check_rate_limit(user_id, "/v1/explain")
        
        features = sanitize_input(transaction.model_dump())
        
        result = await get_cached_or_predict(model, meta or {}, features)
        
        threshold = meta.get("threshold", 0.5) if meta else 0.5
        label = result["fraud_label"]
        prob = result["fraud_probability"]
        
        feature_descriptions = {
            "amount": "High transaction amount increases fraud risk",
            "hour": "Late night transactions are higher risk",
            "device_score": "Low device trust score increases risk",
            "country_risk": "High-risk country contributes to fraud probability",
            "amount_log": "Log-transformed amount reduces outlier impact",
            "is_night": "Nighttime transactions are flagged",
            "high_amount": "Amount exceeds 75th percentile",
            "low_device": "Device reliability is below 0.3",
            "risk_score": "Composite risk score from multiple factors",
            "amount_device_interaction": "Interaction between amount and device score"
        }
        
        feature_cols = get_feature_names()
        df = engineer_features(pd.DataFrame([features]))
        features_eng = df[feature_cols]
        
        contributions = []
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_eng)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            for i, feat in enumerate(feature_cols):
                if i < len(shap_vals):
                    contrib = float(shap_vals[i])
                    contributions.append({
                        "feature": feat,
                        "value": float(features.get(feat, features_eng[feat].values[0])),
                        "contribution": round(contrib, 4),
                        "description": feature_descriptions.get(feat, "")
                    })
            
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            contributions = [{"error": "SHAP calculation failed"}]
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExplanationResponse(
            fraud_probability=float(prob),
            fraud_label=label,
            model_version=meta.get("model_version") if meta else "unknown",
            model_type=meta.get("model_type") if meta else "unknown",
            threshold=threshold,
            prediction_id=prediction_id,
            processing_time_ms=round(processing_time, 2),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            explanation={
                "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, list) else float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.05,
                "feature_contributions": contributions[:10],
                "model_features": feature_cols
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/logs")
async def get_audit_logs(
    action: str = None,
    limit: int = 100,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Get audit logs (admin only)."""
    auth_result = await verify_api_authentication(None, credentials)
    if auth_result["data"].permissions != ["*"] and auth_result["data"].user_id != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    logs = audit_logger.get_logs(action=action)
    return {"logs": logs[-limit:], "total": len(logs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
