import pytest
import json
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


os.environ["API_KEY"] = "test-api-key"


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = MagicMock()
    mock.predict_proba.return_value = [[0.95, 0.05]]
    mock.predict.return_value = [0]
    return mock


@pytest.fixture
def mock_meta():
    """Create mock metadata."""
    return {
        "model_version": "2026-01-16T00:16:33.068343",
        "model_type": "XGBoost",
        "features": ["amount", "hour", "device_score", "country_risk"],
        "threshold": 0.5,
        "target_precision": 0.95,
        "metrics_val": {
            "roc_auc": 0.85,
            "pr_auc": 0.15
        }
    }


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_model_loaded(self, mock_model, mock_meta):
        """Test health check when model is loaded."""
        with patch('src.fraudml.api.app.model', mock_model), \
             patch('src.fraudml.api.app.meta', mock_meta):
            from src.fraudml.api.app import app
            client = TestClient(app)
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is True
            assert "model_version" in data

    def test_health_check_model_not_loaded(self):
        """Test health check when model is not loaded."""
        with patch('src.fraudml.api.app.model', None), \
             patch('src.fraudml.api.app.meta', None):
            from src.fraudml.api.app import app
            client = TestClient(app)
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""

    def test_single_prediction_success(self, mock_model, mock_meta):
        """Test successful single prediction."""
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        with patch('src.fraudml.api.app.model', mock_model), \
             patch('src.fraudml.api.app.meta', mock_meta), \
             patch('src.fraudml.api.app.get_cached_or_predict') as mock_cache, \
             patch('src.fraudml.api.app.audit_logger'):
            
            mock_cache.return_value = {
                "fraud_probability": 0.7,
                "fraud_label": 1,
                "model_version": "test",
                "model_type": "XGBoost",
                "threshold": 0.5
            }
            
            from src.fraudml.api.app import app
            client = TestClient(app)
            
            response = client.post(
                "/v1/predict",
                json={
                    "amount": 1500.00,
                    "hour": 14,
                    "device_score": 0.85,
                    "country_risk": 2
                },
                headers={"Authorization": "Bearer test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "fraud_probability" in data
            assert "fraud_label" in data
            assert "prediction_id" in data
            assert "processing_time_ms" in data

    def test_prediction_invalid_amount(self):
        """Test prediction with invalid amount."""
        from src.fraudml.api.app import app
        client = TestClient(app)
        
        response = client.post(
            "/v1/predict",
            json={
                "amount": -100,
                "hour": 14,
                "device_score": 0.85,
                "country_risk": 2
            },
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 422

    def test_prediction_missing_fields(self):
        """Test prediction with missing required fields."""
        from src.fraudml.api.app import app
        client = TestClient(app)
        
        response = client.post(
            "/v1/predict",
            json={
                "amount": 1500.00
            },
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 422

    def test_prediction_no_auth(self):
        """Test prediction without authentication."""
        from src.fraudml.api.app import app
        client = TestClient(app)
        
        response = client.post(
            "/v1/predict",
            json={
                "amount": 1500.00,
                "hour": 14,
                "device_score": 0.85,
                "country_risk": 2
            }
        )
        
        assert response.status_code in [401, 403]


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_prediction_success(self, mock_model, mock_meta):
        """Test successful batch prediction."""
        mock_model.predict_proba.return_value = [[0.3, 0.7], [0.95, 0.05]]
        
        with patch('src.fraudml.api.app.model', mock_model), \
             patch('src.fraudml.api.app.meta', mock_meta), \
             patch('src.fraudml.api.app.prediction_cache') as mock_cache:
            
            mock_cache.get_many.return_value = {}
            mock_cache.get.return_value = None
            
            from src.fraudml.api.app import app
            client = TestClient(app)
            
            response = client.post(
                "/v1/batch-predict",
                json={
                    "transactions": [
                        {"amount": 100, "hour": 10, "device_score": 0.9, "country_risk": 1},
                        {"amount": 5000, "hour": 3, "device_score": 0.1, "country_risk": 5}
                    ]
                },
                headers={"Authorization": "Bearer test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert "batch_id" in data
            assert "total_processing_time_ms" in data

    def test_batch_prediction_empty(self):
        """Test batch prediction with empty transactions."""
        from src.fraudml.api.app import app
        client = TestClient(app)
        
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": []},
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == []


class TestAuthentication:
    """Tests for authentication."""

    def test_api_key_authentication(self):
        """Test API key authentication."""
        from src.fraudml.api.auth import auth_handler
        
        result = auth_handler.verify_api_key("test-api-key")
        assert result is not None
        assert result.get("role") == "admin"

    def test_invalid_api_key(self):
        """Test invalid API key authentication."""
        from src.fraudml.api.auth import auth_handler
        
        result = auth_handler.verify_api_key("invalid-key")
        assert result is None

    def test_create_access_token(self):
        """Test access token creation."""
        from src.fraudml.api.auth import auth_handler
        
        token, expires = auth_handler.create_access_token("user123", ["predict"])
        
        assert token is not None
        assert expires > 0
        
        payload = auth_handler.verify_token(token)
        assert payload.user_id == "user123"
        assert payload.token_type.value == "access"


class TestCacheStats:
    """Tests for cache statistics endpoint."""

    def test_cache_stats_endpoint(self):
        """Test cache stats endpoint returns valid response."""
        with patch('src.fraudml.api.app.prediction_cache') as mock_cache:
            mock_cache.get_stats.return_value = {
                "total_keys": 100,
                "hits": 500,
                "misses": 100,
                "hit_rate": 0.83
            }
            
            from src.fraudml.api.app import app
            client = TestClient(app)
            
            response = client.get("/cache/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert "total_keys" in data
            assert "hits" in data
            assert "hit_rate" in data


class TestInputValidation:
    """Tests for input validation."""

    def test_amount_range_validation(self):
        """Test amount field validation."""
        from src.fraudml.api.app import TransactionInput
        
        with pytest.raises(ValueError):
            TransactionInput(amount=-100, hour=12, device_score=0.5, country_risk=2)

    def test_hour_range_validation(self):
        """Test hour field validation."""
        from src.fraudml.api.app import TransactionInput
        
        with pytest.raises(ValueError):
            TransactionInput(amount=100, hour=25, device_score=0.5, country_risk=2)

    def test_device_score_range_validation(self):
        """Test device_score field validation."""
        from src.fraudml.api.app import TransactionInput
        
        with pytest.raises(ValueError):
            TransactionInput(amount=100, hour=12, device_score=1.5, country_risk=2)

    def test_country_risk_range_validation(self):
        """Test country_risk field validation."""
        from src.fraudml.api.app import TransactionInput
        
        with pytest.raises(ValueError):
            TransactionInput(amount=100, hour=12, device_score=0.5, country_risk=6)


class TestBatchInputValidation:
    """Tests for batch input validation."""

    def test_batch_size_limit(self):
        """Test batch size limit."""
        from src.fraudml.api.app import BatchTransactionInput, TransactionInput
        
        transactions = [
            TransactionInput(
                amount=100 + i,
                hour=12,
                device_score=0.5,
                country_risk=2
            )
            for i in range(1001)
        ]
        
        with pytest.raises(ValueError):
            BatchTransactionInput(transactions=transactions)
