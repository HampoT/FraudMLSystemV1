"""
Enhanced A/B Router for Fraud Detection System.

Provides:
- Traffic routing based on user_id hash
- Configurable traffic splits (50/50, 80/20, 90/10, etc.)
- Variant tracking and conversion metrics
- Canary deployment support
- Automatic rollback based on metrics
"""
import os
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from threading import Lock

import redis
from redis import ConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class VariantMetrics:
    """Metrics for a model variant."""
    variant_id: str
    total_requests: int = 0
    fraud_predictions: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    avg_fraud_probability: float = 0.0
    last_updated: str = ""
    
    def update(self, latency_ms: float, fraud_label: int, fraud_prob: float, is_error: bool = False):
        """Update metrics with new request data."""
        n = self.total_requests
        self.total_requests += 1
        
        # Rolling average for latency
        self.avg_latency_ms = (self.avg_latency_ms * n + latency_ms) / (n + 1)
        
        # Rolling average for fraud probability
        self.avg_fraud_probability = (self.avg_fraud_probability * n + fraud_prob) / (n + 1)
        
        if fraud_label == 1:
            self.fraud_predictions += 1
        
        if is_error:
            self.error_count += 1
        
        self.last_updated = datetime.utcnow().isoformat()
    
    @property
    def fraud_rate(self) -> float:
        """Calculate fraud prediction rate."""
        return self.fraud_predictions / max(1, self.total_requests)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / max(1, self.total_requests)


@dataclass
class CanaryDeployment:
    """Canary deployment configuration."""
    new_model_version: str
    current_model_version: str
    traffic_percentage: float  # 0.0 to 1.0
    start_time: str
    monitoring_duration_hours: int = 1
    auto_promote: bool = True
    success_criteria: Dict = field(default_factory=lambda: {
        "max_error_rate": 0.05,
        "max_latency_p95_ms": 200,
        "min_requests": 100
    })
    status: str = "active"  # active, promoted, rolled_back
    
    @property
    def is_expired(self) -> bool:
        """Check if monitoring period has expired."""
        start = datetime.fromisoformat(self.start_time)
        return datetime.utcnow() > start + timedelta(hours=self.monitoring_duration_hours)


class ABRouter:
    """A/B test router with traffic splitting and variant tracking."""
    
    def __init__(
        self,
        variants: Dict[str, float] = None,
        redis_url: str = None
    ):
        """Initialize the A/B router.
        
        Args:
            variants: Dict mapping variant_id to traffic weight (0.0-1.0)
                      Weights should sum to 1.0
            redis_url: Redis connection URL for persistent metrics
        """
        self.variants = variants or {"default": 1.0}
        self._validate_weights()
        
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._pool = None
        self._client = None
        
        # In-memory metrics (optionally synced to Redis)
        self.metrics: Dict[str, VariantMetrics] = {}
        self._lock = Lock()
        
        # Initialize metrics for each variant
        for variant_id in self.variants:
            self.metrics[variant_id] = VariantMetrics(variant_id=variant_id)
        
        # Canary deployment state
        self.canary: Optional[CanaryDeployment] = None
    
    def _validate_weights(self):
        """Validate that weights sum to 1.0."""
        total = sum(self.variants.values())
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            self.variants = {k: v / total for k, v in self.variants.items()}
            logger.warning(f"Normalized variant weights to sum to 1.0: {self.variants}")
    
    @property
    def client(self):
        """Lazy Redis client initialization."""
        if self._client is None:
            try:
                if self._pool is None:
                    self._pool = ConnectionPool.from_url(
                        self.redis_url,
                        decode_responses=True
                    )
                self._client = redis.Redis(connection_pool=self._pool)
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._client = None
        return self._client
    
    def route(self, user_id: str) -> str:
        """Route a user to a model variant.
        
        Uses consistent hashing to ensure same user always gets same variant.
        
        Args:
            user_id: User identifier
            
        Returns:
            variant_id for the user
        """
        # Check for canary deployment
        if self.canary and self.canary.status == "active":
            return self._route_with_canary(user_id)
        
        # Standard A/B routing
        return self._route_standard(user_id)
    
    def _route_standard(self, user_id: str) -> str:
        """Standard A/B routing based on user hash."""
        # Hash user_id to get consistent bucket (0.0 - 1.0)
        hash_bytes = hashlib.md5(user_id.encode()).digest()
        bucket = int.from_bytes(hash_bytes[:4], 'big') / (2**32)
        
        # Determine variant based on cumulative weights
        cumulative = 0.0
        for variant_id, weight in self.variants.items():
            cumulative += weight
            if bucket < cumulative:
                return variant_id
        
        # Fallback to first variant
        return list(self.variants.keys())[0]
    
    def _route_with_canary(self, user_id: str) -> str:
        """Route with canary deployment active."""
        # Hash for canary selection
        hash_bytes = hashlib.md5(f"canary:{user_id}".encode()).digest()
        bucket = int.from_bytes(hash_bytes[:4], 'big') / (2**32)
        
        if bucket < self.canary.traffic_percentage:
            return self.canary.new_model_version
        return self.canary.current_model_version
    
    def record_request(
        self,
        variant_id: str,
        latency_ms: float,
        fraud_label: int,
        fraud_prob: float,
        is_error: bool = False
    ):
        """Record metrics for a request.
        
        Args:
            variant_id: Variant that handled the request
            latency_ms: Request latency in milliseconds
            fraud_label: Predicted fraud label (0/1)
            fraud_prob: Predicted fraud probability
            is_error: Whether request resulted in error
        """
        with self._lock:
            if variant_id not in self.metrics:
                self.metrics[variant_id] = VariantMetrics(variant_id=variant_id)
            
            self.metrics[variant_id].update(latency_ms, fraud_label, fraud_prob, is_error)
        
        # Optionally persist to Redis
        self._persist_metrics(variant_id)
        
        # Check canary health
        if self.canary and self.canary.status == "active":
            self._check_canary_health()
    
    def _persist_metrics(self, variant_id: str):
        """Persist metrics to Redis."""
        if self.client:
            try:
                key = f"ab:metrics:{variant_id}"
                self.client.setex(
                    key,
                    3600,  # 1 hour TTL
                    json.dumps(asdict(self.metrics[variant_id]))
                )
            except Exception as e:
                logger.debug(f"Failed to persist metrics: {e}")
    
    def get_metrics(self, variant_id: str = None) -> Dict[str, VariantMetrics]:
        """Get current metrics.
        
        Args:
            variant_id: Specific variant or None for all
            
        Returns:
            Dict of variant_id -> VariantMetrics
        """
        if variant_id:
            return {variant_id: self.metrics.get(variant_id)}
        return self.metrics.copy()
    
    def get_conversion_metrics(self) -> Dict[str, Dict]:
        """Get conversion metrics for all variants.
        
        Returns:
            Dict with per-variant metrics
        """
        result = {}
        for variant_id, metrics in self.metrics.items():
            result[variant_id] = {
                "total_requests": metrics.total_requests,
                "fraud_rate": metrics.fraud_rate,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "avg_fraud_probability": round(metrics.avg_fraud_probability, 4),
            }
        return result
    
    def set_traffic_split(self, variants: Dict[str, float]):
        """Update traffic split configuration.
        
        Args:
            variants: New variant weights
        """
        self.variants = variants
        self._validate_weights()
        
        # Initialize metrics for new variants
        for variant_id in self.variants:
            if variant_id not in self.metrics:
                self.metrics[variant_id] = VariantMetrics(variant_id=variant_id)
        
        logger.info(f"Updated traffic split: {self.variants}")
    
    # Canary deployment methods
    
    def start_canary(
        self,
        new_model_version: str,
        current_model_version: str,
        traffic_percentage: float = 0.1,
        monitoring_hours: int = 1,
        auto_promote: bool = True,
        success_criteria: Dict = None
    ) -> CanaryDeployment:
        """Start a canary deployment.
        
        Args:
            new_model_version: New model to deploy
            current_model_version: Current production model
            traffic_percentage: Traffic to route to new model (0.0-1.0)
            monitoring_hours: How long to monitor before decision
            auto_promote: Whether to auto-promote on success
            success_criteria: Custom success criteria
            
        Returns:
            CanaryDeployment configuration
        """
        self.canary = CanaryDeployment(
            new_model_version=new_model_version,
            current_model_version=current_model_version,
            traffic_percentage=traffic_percentage,
            start_time=datetime.utcnow().isoformat(),
            monitoring_duration_hours=monitoring_hours,
            auto_promote=auto_promote,
            success_criteria=success_criteria or {
                "max_error_rate": 0.05,
                "max_latency_p95_ms": 200,
                "min_requests": 100
            }
        )
        
        # Initialize metrics for canary model
        self.metrics[new_model_version] = VariantMetrics(variant_id=new_model_version)
        
        logger.info(f"Started canary deployment: {traffic_percentage*100}% traffic to {new_model_version}")
        
        # Persist canary state
        self._persist_canary_state()
        
        return self.canary
    
    def _check_canary_health(self):
        """Check canary health and auto-promote/rollback if needed."""
        if not self.canary or self.canary.status != "active":
            return
        
        new_metrics = self.metrics.get(self.canary.new_model_version)
        if not new_metrics:
            return
        
        criteria = self.canary.success_criteria
        
        # Check if enough requests to evaluate
        if new_metrics.total_requests < criteria.get("min_requests", 100):
            return
        
        # Check error rate
        if new_metrics.error_rate > criteria.get("max_error_rate", 0.05):
            logger.warning(f"Canary error rate too high: {new_metrics.error_rate:.2%}")
            if self.canary.auto_promote:
                self.rollback_canary("High error rate")
            return
        
        # Check latency (simple avg as proxy for P95)
        if new_metrics.avg_latency_ms > criteria.get("max_latency_p95_ms", 200):
            logger.warning(f"Canary latency too high: {new_metrics.avg_latency_ms:.2f}ms")
            if self.canary.auto_promote:
                self.rollback_canary("High latency")
            return
        
        # Check if monitoring period expired
        if self.canary.is_expired and self.canary.auto_promote:
            self.promote_canary()
    
    def promote_canary(self) -> bool:
        """Promote canary to full production.
        
        Returns:
            True if promoted
        """
        if not self.canary:
            return False
        
        # Update variant weights to 100% new model
        self.variants = {self.canary.new_model_version: 1.0}
        self.canary.status = "promoted"
        
        logger.info(f"Promoted canary: {self.canary.new_model_version} is now production")
        
        self._persist_canary_state()
        return True
    
    def rollback_canary(self, reason: str = "") -> bool:
        """Rollback canary deployment.
        
        Args:
            reason: Reason for rollback
            
        Returns:
            True if rolled back
        """
        if not self.canary:
            return False
        
        # Revert to current model
        self.variants = {self.canary.current_model_version: 1.0}
        self.canary.status = "rolled_back"
        
        logger.warning(f"Rolled back canary: {reason}")
        
        self._persist_canary_state()
        return True
    
    def get_canary_status(self) -> Optional[Dict]:
        """Get current canary deployment status.
        
        Returns:
            Canary status dict or None
        """
        if not self.canary:
            return None
        
        new_metrics = self.metrics.get(self.canary.new_model_version)
        return {
            "new_model_version": self.canary.new_model_version,
            "current_model_version": self.canary.current_model_version,
            "traffic_percentage": self.canary.traffic_percentage,
            "status": self.canary.status,
            "start_time": self.canary.start_time,
            "is_expired": self.canary.is_expired,
            "metrics": asdict(new_metrics) if new_metrics else None
        }
    
    def _persist_canary_state(self):
        """Persist canary state to Redis."""
        if self.client and self.canary:
            try:
                self.client.set(
                    "ab:canary:state",
                    json.dumps(asdict(self.canary))
                )
            except Exception as e:
                logger.debug(f"Failed to persist canary state: {e}")


# Global router instance
router = ABRouter()


if __name__ == "__main__":
    # Test A/B routing
    print("Testing A/B Router...")
    
    # Create router with 50/50 split
    test_router = ABRouter({"model_a": 0.5, "model_b": 0.5})
    
    # Route some users
    results = defaultdict(int)
    for i in range(100):
        variant = test_router.route(f"user{i}")
        results[variant] += 1
    
    print(f"50/50 split results: {dict(results)}")
    
    # Test consistent routing
    user123_variant1 = test_router.route("user123")
    user123_variant2 = test_router.route("user123")
    print(f"Consistent routing: {user123_variant1} == {user123_variant2}: {user123_variant1 == user123_variant2}")
    
    # Test 80/20 split
    test_router.set_traffic_split({"model_a": 0.8, "model_b": 0.2})
    results = defaultdict(int)
    for i in range(100):
        variant = test_router.route(f"user{i}")
        results[variant] += 1
    
    print(f"80/20 split results: {dict(results)}")
