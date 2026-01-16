import os
import json
import hashlib
import pickle
import redis
from redis import ConnectionPool
import time
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# TTL Tiers (in seconds)
TTL_SHORT = 300      # 5 min - for high-risk predictions
TTL_MEDIUM = 1800    # 30 min - for medium-risk predictions
TTL_LONG = 3600      # 1 hour - for low-risk predictions
TTL_MODEL = 86400    # 24 hours - for model metadata


class PredictionCache:
    """Redis-based caching layer for predictions with connection pooling."""

    def __init__(self, redis_url: str = None, ttl: int = TTL_MEDIUM):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl
        self._pool = None
        self._client = None

    @property
    def pool(self):
        """Lazy initialization of Redis connection pool."""
        if self._pool is None:
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=True
            )
        return self._pool

    @property
    def client(self):
        """Get Redis client from connection pool."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
        return self._client

    def _get_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key from features."""
        features_str = json.dumps(features, sort_keys=True)
        return f"pred:cache:{hashlib.md5(features_str.encode()).hexdigest()}"

    def get(self, features: Dict[str, Any]) -> Optional[Dict]:
        """Get cached prediction.

        Args:
            features: Transaction features

        Returns:
            Cached prediction or None
        """
        try:
            key = self._get_cache_key(features)
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None

    def set(self, features: Dict[str, Any], prediction: Dict, ttl: int = None):
        """Cache a prediction with tiered TTL based on fraud probability.

        Args:
            features: Transaction features
            prediction: Prediction result
            ttl: Time to live in seconds (overrides tier calculation)
        """
        try:
            key = self._get_cache_key(features)
            
            # Calculate tiered TTL if not explicitly provided
            if ttl is None:
                fraud_prob = prediction.get("fraud_probability", 0.5)
                if fraud_prob > 0.7 or fraud_prob < 0.1:
                    # Edge cases: shorter TTL for more volatile predictions
                    ttl = TTL_SHORT
                elif 0.3 < fraud_prob < 0.7:
                    # Uncertain cases: medium TTL
                    ttl = TTL_MEDIUM
                else:
                    # Clear cases: longer TTL
                    ttl = TTL_LONG
            
            self.client.setex(
                key,
                ttl,
                json.dumps(prediction, default=str)
            )
            logger.debug(f"Cached prediction with TTL={ttl}s")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def invalidate(self, features: Dict[str, Any]):
        """Invalidate a cached prediction.

        Args:
            features: Transaction features
        """
        try:
            key = self._get_cache_key(features)
            self.client.delete(key)
        except Exception as e:
            print(f"Cache invalidate error: {e}")

    def clear_all(self):
        """Clear all cached predictions."""
        try:
            keys = self.client.keys("pred:cache:*")
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            print(f"Cache clear error: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            info = self.client.info("stats")
            keys = self.client.keys("pred:cache:*")
            return {
                "total_keys": len(keys),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            }
        except Exception as e:
            return {"error": str(e)}

    def warm_cache(self, common_features: List[Dict[str, Any]], predictor_func):
        """Pre-warm cache with common transaction patterns.
        
        Args:
            common_features: List of common feature combinations to cache
            predictor_func: Function that takes features and returns prediction
        """
        warmed = 0
        for features in common_features:
            try:
                if self.get(features) is None:
                    prediction = predictor_func(features)
                    self.set(features, prediction, ttl=TTL_LONG)
                    warmed += 1
            except Exception as e:
                logger.warning(f"Cache warming failed for features: {e}")
        
        logger.info(f"Cache warming complete: {warmed}/{len(common_features)} entries added")
        return warmed


class ModelCache:
    """Cache for model loading."""

    def __init__(self):
        self._model = None
        self._meta = None
        self._last_load = 0
        self._load_interval = 300

    def get_model(self, model_path: str, meta_path: str):
        """Get model with caching.

        Args:
            model_path: Path to model file
            meta_path: Path to metadata file

        Returns:
            Tuple of (model, metadata)
        """
        current_time = time.time()

        if (self._model is None or
            current_time - self._last_load > self._load_interval or
            not os.path.exists(model_path)):

            if os.path.exists(model_path):
                self._model = joblib.load(model_path)
                with open(meta_path, 'r') as f:
                    self._meta = json.load(f)
                self._last_load = current_time

        return self._model, self._meta

    def clear(self):
        """Clear cached model."""
        self._model = None
        self._meta = None
        self._last_load = 0


class FeatureCache:
    """Cache for computed features."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def get_features(self, user_id: str, merchant_id: str = None) -> Dict:
        """Get cached features for user/merchant.

        Args:
            user_id: User identifier
            merchant_id: Merchant identifier

        Returns:
            Dictionary of features
        """
        try:
            key = f"features:user:{user_id}"
            if merchant_id:
                key += f":merch:{merchant_id}"
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Feature cache get error: {e}")
        return {}

    def set_features(self, user_id: str, features: Dict, ttl: int = 3600):
        """Cache features for user.

        Args:
            user_id: User identifier
            features: Feature dictionary
            ttl: Time to live
        """
        try:
            key = f"features:user:{user_id}"
            self.client.setex(key, ttl, json.dumps(features))
        except Exception as e:
            print(f"Feature cache set error: {e}")


prediction_cache = PredictionCache()
model_cache = ModelCache()
feature_cache = FeatureCache()
