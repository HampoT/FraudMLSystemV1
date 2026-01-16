"""
Feature Store for Fraud Detection System.

Provides centralized feature storage and retrieval with:
- Redis backend for fast access
- Feature versioning
- Feature reuse between training and inference
- Feature metadata and lineage tracking
"""
import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict

import redis
from redis import ConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a stored feature."""
    name: str
    version: str
    created_at: str
    updated_at: str
    dtype: str
    source: str = "computed"
    description: str = ""


class FeatureStore:
    """Redis-based feature store with versioning support.
    
    Provides get/set operations for features with:
    - User/entity-level feature storage
    - Feature versioning
    - TTL management
    - Feature metadata
    """
    
    DEFAULT_VERSION = "v1"
    DEFAULT_TTL = 86400  # 24 hours
    
    def __init__(
        self,
        redis_url: str = None,
        default_ttl: int = None,
        prefix: str = "fs"
    ):
        """Initialize the feature store.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL for features in seconds
            prefix: Key prefix for feature store
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.default_ttl = default_ttl or self.DEFAULT_TTL
        self.prefix = prefix
        self._pool = None
        self._client = None
    
    @property
    def pool(self):
        """Lazy initialization of Redis connection pool."""
        if self._pool is None:
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
                decode_responses=True
            )
        return self._pool
    
    @property
    def client(self):
        """Get Redis client from connection pool."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
        return self._client
    
    def _make_key(
        self,
        entity_id: str,
        feature_name: str,
        version: str = None
    ) -> str:
        """Create Redis key for a feature.
        
        Args:
            entity_id: User or entity identifier
            feature_name: Name of the feature
            version: Feature version (optional)
            
        Returns:
            Redis key string
        """
        version = version or self.DEFAULT_VERSION
        return f"{self.prefix}:{version}:{entity_id}:{feature_name}"
    
    def _make_metadata_key(self, feature_name: str, version: str = None) -> str:
        """Create Redis key for feature metadata."""
        version = version or self.DEFAULT_VERSION
        return f"{self.prefix}:meta:{version}:{feature_name}"
    
    def set_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        version: str = None,
        ttl: int = None,
        metadata: Dict = None
    ) -> bool:
        """Store a feature value.
        
        Args:
            entity_id: User or entity identifier
            feature_name: Name of the feature
            value: Feature value (serializable)
            version: Feature version
            ttl: Time to live in seconds
            metadata: Optional metadata dict
            
        Returns:
            True if successful
        """
        try:
            key = self._make_key(entity_id, feature_name, version)
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value)
            else:
                serialized = str(value)
            
            # Store with TTL
            self.client.setex(key, ttl or self.default_ttl, serialized)
            
            # Update metadata if provided
            if metadata:
                self._update_metadata(feature_name, version, metadata)
            
            logger.debug(f"Set feature {feature_name} for {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set feature: {e}")
            return False
    
    def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        version: str = None,
        default: Any = None
    ) -> Any:
        """Retrieve a feature value.
        
        Args:
            entity_id: User or entity identifier
            feature_name: Name of the feature
            version: Feature version
            default: Default value if not found
            
        Returns:
            Feature value or default
        """
        try:
            key = self._make_key(entity_id, feature_name, version)
            value = self.client.get(key)
            
            if value is None:
                return default
            
            # Try to parse as JSON, fall back to numeric/string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    return float(value)
                except ValueError:
                    return value
                    
        except Exception as e:
            logger.error(f"Failed to get feature: {e}")
            return default
    
    def set_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        version: str = None,
        ttl: int = None
    ) -> bool:
        """Store multiple features for an entity.
        
        Args:
            entity_id: User or entity identifier
            features: Dict of feature_name -> value
            version: Feature version
            ttl: Time to live in seconds
            
        Returns:
            True if all successful
        """
        try:
            pipeline = self.client.pipeline()
            
            for feature_name, value in features.items():
                key = self._make_key(entity_id, feature_name, version)
                if isinstance(value, (dict, list)):
                    serialized = json.dumps(value)
                else:
                    serialized = str(value)
                pipeline.setex(key, ttl or self.default_ttl, serialized)
            
            pipeline.execute()
            logger.debug(f"Set {len(features)} features for {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set features: {e}")
            return False
    
    def get_features(
        self,
        entity_id: str,
        feature_names: List[str],
        version: str = None
    ) -> Dict[str, Any]:
        """Retrieve multiple features for an entity.
        
        Args:
            entity_id: User or entity identifier
            feature_names: List of feature names
            version: Feature version
            
        Returns:
            Dict of feature_name -> value (missing features not included)
        """
        try:
            pipeline = self.client.pipeline()
            
            keys = [self._make_key(entity_id, name, version) for name in feature_names]
            for key in keys:
                pipeline.get(key)
            
            values = pipeline.execute()
            
            result = {}
            for name, value in zip(feature_names, values):
                if value is not None:
                    try:
                        result[name] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        try:
                            result[name] = float(value)
                        except ValueError:
                            result[name] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get features: {e}")
            return {}
    
    def delete_feature(
        self,
        entity_id: str,
        feature_name: str,
        version: str = None
    ) -> bool:
        """Delete a feature.
        
        Args:
            entity_id: User or entity identifier
            feature_name: Name of the feature
            version: Feature version
            
        Returns:
            True if deleted
        """
        try:
            key = self._make_key(entity_id, feature_name, version)
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete feature: {e}")
            return False
    
    def _update_metadata(
        self,
        feature_name: str,
        version: str,
        metadata: Dict
    ):
        """Update feature metadata."""
        key = self._make_metadata_key(feature_name, version)
        now = datetime.utcnow().isoformat()
        
        existing = self.client.get(key)
        if existing:
            meta = json.loads(existing)
            meta.update(metadata)
            meta["updated_at"] = now
        else:
            meta = {
                "name": feature_name,
                "version": version or self.DEFAULT_VERSION,
                "created_at": now,
                "updated_at": now,
                **metadata
            }
        
        self.client.set(key, json.dumps(meta))
    
    def get_metadata(
        self,
        feature_name: str,
        version: str = None
    ) -> Optional[Dict]:
        """Get feature metadata.
        
        Args:
            feature_name: Name of the feature
            version: Feature version
            
        Returns:
            Metadata dict or None
        """
        try:
            key = self._make_metadata_key(feature_name, version)
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None
    
    def get_all_features_for_entity(
        self,
        entity_id: str,
        version: str = None
    ) -> Dict[str, Any]:
        """Get all features for an entity.
        
        Args:
            entity_id: User or entity identifier
            version: Feature version
            
        Returns:
            Dict of all features for the entity
        """
        try:
            version = version or self.DEFAULT_VERSION
            pattern = f"{self.prefix}:{version}:{entity_id}:*"
            
            result = {}
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                
                if keys:
                    values = self.client.mget(keys)
                    for key, value in zip(keys, values):
                        if value:
                            feature_name = key.split(":")[-1]
                            try:
                                result[feature_name] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                try:
                                    result[feature_name] = float(value)
                                except ValueError:
                                    result[feature_name] = value
                
                if cursor == 0:
                    break
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get all features: {e}")
            return {}
    
    def compute_and_store(
        self,
        entity_id: str,
        raw_data: Dict[str, Any],
        version: str = None,
        ttl: int = None
    ) -> Dict[str, Any]:
        """Compute features from raw data and store them.
        
        Args:
            entity_id: User or entity identifier
            raw_data: Raw transaction data
            version: Feature version
            ttl: Time to live
            
        Returns:
            Dict of computed features
        """
        import pandas as pd
        import numpy as np
        
        # Compute features
        df = pd.DataFrame([raw_data])
        
        features = {
            "amount": raw_data.get("amount", 0),
            "hour": raw_data.get("hour", 0),
            "device_score": raw_data.get("device_score", 0.5),
            "country_risk": raw_data.get("country_risk", 1),
            "amount_log": float(np.log1p(raw_data.get("amount", 0))),
            "is_night": int(0 <= raw_data.get("hour", 12) <= 5),
            "is_evening": int(18 <= raw_data.get("hour", 12) <= 23),
            "low_device": int(raw_data.get("device_score", 0.5) < 0.3),
            "high_risk_country": int(raw_data.get("country_risk", 1) >= 4),
        }
        
        # Compute risk score
        features["risk_score"] = (
            features.get("low_device", 0) * 0.3 +
            features.get("high_risk_country", 0) * 0.2 +
            features.get("is_night", 0) * 0.2
        )
        
        # Store features
        self.set_features(entity_id, features, version, ttl)
        
        return features
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics.
        
        Returns:
            Dict with store statistics
        """
        try:
            info = self.client.info()
            pattern = f"{self.prefix}:*"
            
            # Count keys
            cursor = 0
            key_count = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=1000)
                key_count += len(keys)
                if cursor == 0:
                    break
            
            return {
                "total_features": key_count,
                "memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "redis_version": info.get("redis_version", "unknown")
            }
        except Exception as e:
            return {"error": str(e)}


# Global feature store instance
store = FeatureStore()


# Convenience functions
def get_feature(entity_id: str, feature_name: str, version: str = None) -> Any:
    """Get a single feature value."""
    return store.get_feature(entity_id, feature_name, version)


def set_feature(entity_id: str, feature_name: str, value: Any, version: str = None) -> bool:
    """Set a single feature value."""
    return store.set_feature(entity_id, feature_name, value, version)


def get_features(entity_id: str, feature_names: List[str], version: str = None) -> Dict[str, Any]:
    """Get multiple feature values."""
    return store.get_features(entity_id, feature_names, version)


def set_features(entity_id: str, features: Dict[str, Any], version: str = None) -> bool:
    """Set multiple feature values."""
    return store.set_features(entity_id, features, version)


if __name__ == "__main__":
    # Test the feature store
    print("Testing Feature Store...")
    
    # Set a feature
    success = store.set_feature("user123", "avg_amount", 500)
    print(f"Set feature: {success}")
    
    # Get the feature
    value = store.get_feature("user123", "avg_amount")
    print(f"Get feature: {value}")
    
    # Set multiple features
    store.set_features("user123", {
        "total_transactions": 100,
        "fraud_rate": 0.05,
        "last_amount": 250.0
    })
    
    # Get multiple features
    features = store.get_features("user123", ["avg_amount", "total_transactions", "fraud_rate"])
    print(f"Get features: {features}")
    
    # Get stats
    stats = store.get_stats()
    print(f"Stats: {stats}")
