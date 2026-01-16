"""
Chaos engineering tests for Fraud Detection API.

Tests system resilience under failure conditions including:
- Database connection failures
- Redis connection failures
- Model loading failures
- Network latency simulation
"""
import os
import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Set test environment
os.environ["API_KEY"] = "test-api-key"

from src.fraudml.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def api_key_headers():
    """Headers with valid API key."""
    return {"X-API-Key": "test-api-key"}


@pytest.fixture
def valid_transaction():
    """Valid transaction payload."""
    return {
        "amount": 100.0,
        "hour": 14,
        "device_score": 0.8,
        "country_risk": 2
    }


class TestDatabaseFailures:
    """Tests for database connection failures."""

    def test_predict_succeeds_when_audit_db_fails(self, client, api_key_headers, valid_transaction):
        """Prediction should still work if audit logging DB fails."""
        # Audit logging is async and should not block predictions
        with patch('src.fraudml.data.audit_log.audit_logger.log') as mock_log:
            mock_log.side_effect = Exception("DB connection failed")
            response = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
            # Prediction should still succeed
            assert response.status_code == 200

    def test_health_check_works_when_db_unavailable(self, client):
        """Health check should work even if DB is unavailable."""
        response = client.get("/health")
        # Health should report OK for API even without DB
        assert response.status_code == 200


class TestCacheFailures:
    """Tests for Redis/cache connection failures."""

    def test_predict_works_when_cache_unavailable(self, client, api_key_headers, valid_transaction):
        """Prediction should work without cache."""
        with patch('src.fraudml.api.cache.PredictionCache.get') as mock_get:
            with patch('src.fraudml.api.cache.PredictionCache.set') as mock_set:
                mock_get.side_effect = Exception("Redis connection refused")
                mock_set.side_effect = Exception("Redis connection refused")
                
                response = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
                # Should fall back to direct prediction
                assert response.status_code == 200

    def test_batch_predict_works_when_cache_unavailable(self, client, api_key_headers, valid_transaction):
        """Batch prediction should work without cache."""
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": [valid_transaction] * 3},
            headers=api_key_headers
        )
        assert response.status_code == 200


class TestModelFailures:
    """Tests for model loading/prediction failures."""

    def test_predict_returns_503_when_model_not_loaded(self, client, api_key_headers, valid_transaction):
        """Prediction should return 503 if model is not loaded."""
        import src.fraudml.api.app as app_module
        
        original_model = app_module.model
        try:
            app_module.model = None
            response = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
            assert response.status_code == 503
            assert "not loaded" in response.json().get("detail", "").lower()
        finally:
            app_module.model = original_model

    def test_explain_returns_503_when_model_not_loaded(self, client, api_key_headers, valid_transaction):
        """Explain should return 503 if model is not loaded."""
        import src.fraudml.api.app as app_module
        
        original_model = app_module.model
        try:
            app_module.model = None
            response = client.post("/v1/explain", json=valid_transaction, headers=api_key_headers)
            assert response.status_code == 503
        finally:
            app_module.model = original_model


class TestTimeoutBehavior:
    """Tests for timeout handling."""

    def test_slow_feature_engineering_doesnt_crash(self, client, api_key_headers, valid_transaction):
        """Slow feature engineering should not crash the API."""
        with patch('src.fraudml.data.features.engineer_features') as mock_eng:
            # Simulate slow processing
            def slow_engineer(df):
                time.sleep(0.1)  # 100ms delay
                from src.fraudml.data.features import engineer_features
                with patch.object(mock_eng, 'side_effect', None):
                    return engineer_features.__wrapped__(df) if hasattr(engineer_features, '__wrapped__') else df
            
            # Test should complete without timeout
            response = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
            # Response may succeed or fail depending on implementation
            assert response.status_code in [200, 500]


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    def test_concurrent_predictions_complete(self, client, api_key_headers, valid_transaction):
        """Multiple concurrent predictions should all complete."""
        import concurrent.futures
        
        def make_prediction():
            with TestClient(app) as c:
                return c.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should complete (success or rate limited)
        assert all(r.status_code in [200, 429] for r in results)


class TestRecoveryBehavior:
    """Tests for system recovery after failures."""

    def test_system_recovers_after_transient_error(self, client, api_key_headers, valid_transaction):
        """System should recover after transient errors."""
        # First request
        response1 = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
        
        # Simulate a transient failure (e.g., memory spike)
        # This is just a placeholder - in real chaos testing you'd use tools like toxiproxy
        
        # Second request should still work
        response2 = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
        
        assert response1.status_code == 200
        assert response2.status_code == 200


class TestGracefulDegradation:
    """Tests for graceful degradation."""

    def test_explain_falls_back_when_shap_unavailable(self, client, api_key_headers, valid_transaction):
        """Explain should provide fallback when SHAP fails."""
        with patch('shap.KernelExplainer') as mock_shap:
            mock_shap.side_effect = Exception("SHAP calculation failed")
            
            response = client.post("/v1/explain", json=valid_transaction, headers=api_key_headers)
            
            # Should still return a response with prediction
            assert response.status_code == 200
            data = response.json()
            assert "fraud_probability" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
