import os
import json
import hashlib
import pickle
import redis
import time
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class AsyncPredictionCache:
    """Redis-based async caching layer for predictions."""

    def __init__(self, redis_url: str = None, ttl: int = 3600):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl
        self._client = None
        self._lock = asyncio.Lock()

    async def get_client(self):
        """Get or create Redis client asynchronously."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = redis.Redis.from_url(
                        self.redis_url,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5
                    )
        return self._client

    def _get_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key from features."""
        features_str = json.dumps(features, sort_keys=True, default=str)
        return f"pred:cache:{hashlib.md5(features_str.encode()).hexdigest()}"

    async def get(self, features: Dict[str, Any]) -> Optional[Dict]:
        """Get cached prediction asynchronously."""
        try:
            client = await self.get_client()
            key = self._get_cache_key(features)
            cached = await asyncio.to_thread(client.get, key)
            if cached:
                result = json.loads(cached)
                result["cache_hit"] = True
                return result
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None

    async def set(self, features: Dict[str, Any], prediction: Dict, ttl: int = None):
        """Cache a prediction asynchronously."""
        try:
            client = await self.get_client()
            key = self._get_cache_key(features)
            await asyncio.to_thread(
                client.setex,
                key,
                ttl or self.ttl,
                json.dumps(prediction, default=str)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def get_many(self, features_list: List[Dict[str, Any]]) -> Dict[str, Optional[Dict]]:
        """Get multiple cached predictions."""
        results = {}
        try:
            client = await self.get_client()
            keys = [self._get_cache_key(f) for f in features_list]
            if keys:
                cached = await asyncio.to_thread(client.mget, keys)
                for features, value in zip(features_list, cached):
                    if value:
                        result = json.loads(value)
                        result["cache_hit"] = True
                        results[self._get_cache_key(features)] = result
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
        return results

    async def set_many(self, predictions: Dict[Dict[str, Any], Dict], ttl: int = None):
        """Set multiple predictions."""
        try:
            client = await self.get_client()
            pipeline = client.pipeline()
            for features, prediction in predictions.items():
                key = self._get_cache_key(features)
                pipeline.setex(key, ttl or self.ttl, json.dumps(prediction, default=str))
            await asyncio.to_thread(pipeline.execute)
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")

    async def invalidate(self, features: Dict[str, Any]):
        """Invalidate a cached prediction."""
        try:
            client = await self.get_client()
            key = self._get_cache_key(features)
            await asyncio.to_thread(client.delete, key)
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")

    async def clear_all(self):
        """Clear all cached predictions."""
        try:
            client = await self.get_client()
            keys = await asyncio.to_thread(client.keys, "pred:cache:*")
            if keys:
                await asyncio.to_thread(client.delete, *keys)
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    async def get_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            client = await self.get_client()
            info = await asyncio.to_thread(client.info, "stats")
            keys = await asyncio.to_thread(client.keys, "pred:cache:*")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            return {
                "total_keys": len(keys),
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / max(1, hits + misses)
            }
        except Exception as e:
            return {"error": str(e)}

    async def warm_cache(self, sample_transactions: List[Dict[str, Any]], model, meta: Dict):
        """Pre-warm cache with sample transactions."""
        logger.info(f"Warming cache with {len(sample_transactions)} transactions")
        import pandas as pd
        from ..data.features import engineer_features, get_feature_names
        
        cache_warmed = 0
        
        for tx in sample_transactions:
            try:
                features = {
                    "amount": tx.get("amount", 100),
                    "hour": tx.get("hour", 12),
                    "device_score": tx.get("device_score", 0.5),
                    "country_risk": tx.get("country_risk", 2)
                }
                
                existing = await self.get(features)
                if existing is None:
                    feature_cols = get_feature_names()
                    df = engineer_features(pd.DataFrame([features]))
                    features_eng = df[feature_cols]
                    
                    prob = model.predict_proba(features_eng)[0, 1]
                    threshold = meta.get("threshold", 0.5)
                    label = int(prob >= threshold)
                    
                    prediction = {
                        "fraud_probability": float(prob),
                        "fraud_label": label,
                        "model_version": meta.get("model_version"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await self.set(features, prediction, ttl=7200)
                    cache_warmed += 1
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
        
        logger.info(f"Cache warming complete: {cache_warmed} entries cached")
        return cache_warmed


class ModelCache:
    """Thread-safe model caching with lazy loading."""

    def __init__(self):
        self._model = None
        self._meta = None
        self._last_load = 0
        self._load_interval = 300
        self._lock = asyncio.Lock()

    async def get_model(self, model_path: str, meta_path: str):
        """Get model with async lazy loading."""
        current_time = time.time()

        if (self._model is None or
            current_time - self._last_load > self._load_interval or
            not os.path.exists(model_path)):

            async with self._lock:
                if (self._model is None or
                    current_time - self._last_load > self._load_interval or
                    not os.path.exists(model_path)):
                    
                    if os.path.exists(model_path):
                        import joblib
                        self._model = joblib.load(model_path)
                        with open(meta_path, 'r') as f:
                            self._meta = json.load(f)
                        self._last_load = current_time
                        logger.info("Model loaded into cache")

        return self._model, self._meta

    async def clear(self):
        """Clear cached model."""
        async with self._lock:
            self._model = None
            self._meta = None
            self._last_load = 0


class FeatureCache:
    """Cache for computed user/merchant features."""

    def __init__(self, redis_url: str = None, ttl: int = 3600):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl
        self._client = None
        self._lock = asyncio.Lock()

    async def get_client(self):
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def get_features(self, user_id: str, merchant_id: str = None) -> Dict:
        """Get cached features for user/merchant."""
        try:
            client = await self.get_client()
            key = f"features:user:{user_id}"
            if merchant_id:
                key += f":merch:{merchant_id}"
            cached = await asyncio.to_thread(client.get, key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Feature cache get error: {e}")
        return {}

    async def set_features(self, user_id: str, features: Dict, ttl: int = None):
        """Cache features for user."""
        try:
            client = await self.get_client()
            key = f"features:user:{user_id}"
            await asyncio.to_thread(client.setex, key, ttl or self.ttl, json.dumps(features))
        except Exception as e:
            logger.error(f"Feature cache set error: {e}")


class RequestQueue:
    """Async request queue for rate limiting and processing."""

    def __init__(self, max_concurrent: int = 100):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_times: Dict[str, List[float]] = {}

    async def process(self, user_id: str, operation, *args, **kwargs):
        """Process operation with concurrency control."""
        async with self.semaphore:
            return await operation(*args, **kwargs)

    def record_request(self, user_id: str):
        """Record a request timestamp for rate limiting."""
        now = time.time()
        if user_id not in self.request_times:
            self.request_times[user_id] = []
        self.request_times[user_id].append(now)

    def get_request_count(self, user_id: str, window: int = 60) -> int:
        """Get request count in time window."""
        now = time.time()
        if user_id not in self.request_times:
            return 0
        return sum(
            1 for t in self.request_times[user_id]
            if now - t < window
        )

    def cleanup_old_requests(self, window: int = 60):
        """Clean up old request timestamps."""
        now = time.time()
        for user_id in list(self.request_times.keys()):
            self.request_times[user_id] = [
                t for t in self.request_times[user_id]
                if now - t < window
            ]
            if not self.request_times[user_id]:
                del self.request_times[user_id]


prediction_cache = AsyncPredictionCache()
model_cache = ModelCache()
feature_cache = FeatureCache()
request_queue = RequestQueue(max_concurrent=100)


async def get_cached_or_predict(model, meta: Dict, features: Dict) -> Dict:
    """Get cached prediction or compute and cache."""
    import pandas as pd
    from ..data.features import engineer_features, get_feature_names
    
    cached = await prediction_cache.get(features)
    if cached:
        return cached

    feature_cols = get_feature_names()
    df = engineer_features(pd.DataFrame([features]))
    features_eng = df[feature_cols]

    prob = model.predict_proba(features_eng)[0, 1]
    threshold = meta.get("threshold", 0.5)
    label = int(prob >= threshold)

    prediction = {
        "fraud_probability": float(prob),
        "fraud_label": label,
        "model_version": meta.get("model_version"),
        "model_type": meta.get("model_type"),
        "threshold": threshold,
        "timestamp": datetime.utcnow().isoformat()
    }

    await prediction_cache.set(features, prediction)
    return prediction
