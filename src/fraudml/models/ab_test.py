import os
import random
import hashlib
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ModelVersion(Enum):
    A = "model_a"
    B = "model_b"


@dataclass
class ABTestConfig:
    model_a_name: str
    model_b_name: str
    traffic_split_a: float
    traffic_split_b: float
    min_sample_size: int
    test_duration_days: int


class ABTestRouter:
    """Router for A/B testing between model versions."""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self._traffic_split = config.traffic_split_a / 100.0

    def get_model_for_request(self, user_id: str = None,
                               session_id: str = None) -> Tuple[str, Dict]:
        """Determine which model to use for a request.

        Args:
            user_id: Optional user identifier for consistent routing
            session_id: Optional session identifier

        Returns:
            Tuple of (model_name, routing_metadata)
        """
        if user_id:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            rand_val = (hash_val % 100) / 100.0
        elif session_id:
            hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            rand_val = (hash_val % 100) / 100.0
        else:
            rand_val = random.random()

        if rand_val < self._traffic_split:
            model = self.config.model_a_name
            variant = "A"
        else:
            model = self.config.model_b_name
            variant = "B"

        metadata = {
            "variant": variant,
            "traffic_split": {
                "a": self.config.traffic_split_a,
                "b": self.config.traffic_split_b
            },
            "routing_method": "user_hash" if user_id else "random"
        }

        return model, metadata

    def track_conversion(self, prediction_id: str, variant: str,
                         outcome: bool, metadata: Dict = None):
        """Track conversion outcome for A/B test analysis.

        Args:
            prediction_id: ID of the prediction
            variant: Model variant used (A or B)
            outcome: Whether conversion occurred
            metadata: Additional metadata
        """
        log_entry = {
            "prediction_id": prediction_id,
            "variant": variant,
            "outcome": outcome,
            "metadata": metadata or {}
        }
        print(f"A/B Test Log: {log_entry}")


class BanditRouter:
    """Multi-armed bandit for adaptive model selection."""

    def __init__(self, models: Dict[str, float], exploration_rate: float = 0.1):
        self.models = models
        self.exploration_rate = exploration_rate
        self._counts = {m: 0 for m in models}
        self._rewards = {m: 0.0 for m in models}

    def select_model(self) -> str:
        """Select model using epsilon-greedy strategy."""
        if random.random() < self.exploration_rate:
            return random.choice(list(self.models.keys()))
        else:
            avg_rewards = {m: self._rewards[m] / max(1, self._counts[m])
                          for m in self.models}
            return max(avg_rewards, key=avg_rewards.get)

    def update(self, model: str, reward: float):
        """Update model statistics with new reward."""
        self._counts[model] += 1
        self._rewards[model] += reward

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "counts": self._counts,
            "rewards": self._rewards,
            "avg_rewards": {
                m: self._rewards[m] / max(1, self._counts[m])
                for m in self.models
            }
        }


def create_default_ab_config() -> ABTestConfig:
    """Create default A/B test configuration."""
    return ABTestConfig(
        model_a_name="FraudDetection_LogisticRegression",
        model_b_name="FraudDetection_XGBoost",
        traffic_split_a=50,
        traffic_split_b=50,
        min_sample_size=1000,
        test_duration_days=7
    )


def canary_deploy(current_version: str, new_version: str,
                  canary_percent: int = 10) -> str:
    """Determine which version to route to for canary deployment.

    Args:
        current_version: Current production model
        new_version: New model being tested
        canary_percent: Percentage of traffic for canary

    Returns:
        Model version to use
    """
    rand = random.randint(1, 100)
    if rand <= canary_percent:
        return new_version
    return current_version
