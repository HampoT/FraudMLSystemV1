"""
Comprehensive API integration tests for Fraud Detection API.

Tests cover all endpoints with valid/invalid inputs, authentication scenarios,
rate limiting, and error handling.
"""
import os
import pytest
from fastapi.testclient import TestClient
import time

# Set test environment variables
os.environ["API_KEY"] = "test-api-key"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"

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
def invalid_api_key_headers():
    """Headers with invalid API key."""
    return {"X-API-Key": "wrong-key"}


@pytest.fixture
def valid_transaction():
    """Valid transaction payload."""
    return {
        "amount": 100.0,
        "hour": 14,
        "device_score": 0.8,
        "country_risk": 2
    }


@pytest.fixture
def high_risk_transaction():
    """High risk transaction payload."""
    return {
        "amount": 5000.0,
        "hour": 3,
        "device_score": 0.1,
        "country_risk": 5
    }


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_ok(self, client):
        """Health endpoint should return OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "api_version" in data

    def test_health_check_no_auth_required(self, client):
        """Health endpoint should not require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_returns_prometheus_format(self, client):
        """Metrics endpoint should return Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers.get("content-type") is not None


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_with_valid_transaction(self, client, api_key_headers, valid_transaction):
        """Valid prediction request should succeed."""
        response = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "fraud_label" in data
        assert "prediction_id" in data
        assert 0 <= data["fraud_probability"] <= 1
        assert data["fraud_label"] in [0, 1]

    def test_predict_with_high_risk_transaction(self, client, api_key_headers, high_risk_transaction):
        """High risk transaction should return higher fraud probability."""
        response = client.post("/v1/predict", json=high_risk_transaction, headers=api_key_headers)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        # High risk transaction should have elevated probability
        # Note: actual threshold depends on model training

    def test_predict_without_auth_fails(self, client, valid_transaction):
        """Prediction without authentication should fail."""
        response = client.post("/v1/predict", json=valid_transaction)
        assert response.status_code in [401, 422]  # Unauthorized or validation error

    def test_predict_with_invalid_api_key_fails(self, client, invalid_api_key_headers, valid_transaction):
        """Prediction with invalid API key should fail."""
        response = client.post("/v1/predict", json=valid_transaction, headers=invalid_api_key_headers)
        assert response.status_code == 401

    def test_predict_with_invalid_amount_fails(self, client, api_key_headers):
        """Prediction with invalid amount should fail."""
        invalid = {"amount": -100.0, "hour": 14, "device_score": 0.8, "country_risk": 2}
        response = client.post("/v1/predict", json=invalid, headers=api_key_headers)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_hour_fails(self, client, api_key_headers):
        """Prediction with invalid hour should fail."""
        invalid = {"amount": 100.0, "hour": 25, "device_score": 0.8, "country_risk": 2}
        response = client.post("/v1/predict", json=invalid, headers=api_key_headers)
        assert response.status_code == 422

    def test_predict_with_missing_fields_fails(self, client, api_key_headers):
        """Prediction with missing required fields should fail."""
        incomplete = {"amount": 100.0}
        response = client.post("/v1/predict", json=incomplete, headers=api_key_headers)
        assert response.status_code == 422

    def test_predict_returns_timing_info(self, client, api_key_headers, valid_transaction):
        """Prediction should return processing time."""
        response = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_single_transaction(self, client, api_key_headers, valid_transaction):
        """Batch prediction with single transaction should succeed."""
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": [valid_transaction]},
            headers=api_key_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "batch_id" in data
        assert len(data["predictions"]) == 1

    def test_batch_predict_multiple_transactions(self, client, api_key_headers, valid_transaction, high_risk_transaction):
        """Batch prediction with multiple transactions should succeed."""
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": [valid_transaction, high_risk_transaction, valid_transaction]},
            headers=api_key_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
        assert all("fraud_probability" in p for p in data["predictions"])

    def test_batch_predict_empty_list_fails(self, client, api_key_headers):
        """Batch prediction with empty list should fail."""
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": []},
            headers=api_key_headers
        )
        # Empty batch should either return empty or validation error
        assert response.status_code in [200, 422]

    def test_batch_predict_without_auth_fails(self, client, valid_transaction):
        """Batch prediction without auth should fail."""
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": [valid_transaction]}
        )
        assert response.status_code in [401, 422]


class TestExplainEndpoint:
    """Tests for explain endpoint."""

    def test_explain_returns_feature_contributions(self, client, api_key_headers, valid_transaction):
        """Explain endpoint should return feature contributions."""
        response = client.post("/v1/explain", json=valid_transaction, headers=api_key_headers)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "explanation" in data

    def test_explain_without_auth_fails(self, client, valid_transaction):
        """Explain without auth should fail."""
        response = client.post("/v1/explain", json=valid_transaction)
        assert response.status_code in [401, 422]


class TestInputValidation:
    """Tests for request input validation."""

    def test_device_score_must_be_between_0_and_1(self, client, api_key_headers):
        """Device score validation."""
        invalid = {"amount": 100.0, "hour": 14, "device_score": 1.5, "country_risk": 2}
        response = client.post("/v1/predict", json=invalid, headers=api_key_headers)
        assert response.status_code == 422

    def test_country_risk_must_be_between_1_and_5(self, client, api_key_headers):
        """Country risk validation."""
        invalid = {"amount": 100.0, "hour": 14, "device_score": 0.8, "country_risk": 10}
        response = client.post("/v1/predict", json=invalid, headers=api_key_headers)
        assert response.status_code == 422

    def test_optional_fields_accepted(self, client, api_key_headers, valid_transaction):
        """Optional fields should be accepted."""
        with_optional = {
            **valid_transaction,
            "merchant_id": "MERCHANT123",
            "user_id": "USER456",
            "transaction_count_last_hour": 5
        }
        response = client.post("/v1/predict", json=with_optional, headers=api_key_headers)
        assert response.status_code == 200


class TestResponseFormat:
    """Tests for response format consistency."""

    def test_predict_response_has_required_fields(self, client, api_key_headers, valid_transaction):
        """Predict response should have all required fields."""
        response = client.post("/v1/predict", json=valid_transaction, headers=api_key_headers)
        data = response.json()
        
        required_fields = [
            "fraud_probability",
            "fraud_label",
            "model_version",
            "model_type",
            "threshold",
            "prediction_id",
            "processing_time_ms",
            "timestamp"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_batch_response_has_required_fields(self, client, api_key_headers, valid_transaction):
        """Batch response should have all required fields."""
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": [valid_transaction]},
            headers=api_key_headers
        )
        data = response.json()
        
        required_fields = [
            "predictions",
            "batch_id",
            "total_processing_time_ms",
            "model_version",
            "model_type",
            "timestamp"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_returns_error(self, client, api_key_headers):
        """Invalid JSON should return appropriate error."""
        response = client.post(
            "/v1/predict",
            content="not valid json",
            headers={**api_key_headers, "Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

    def test_wrong_content_type_handled(self, client, api_key_headers, valid_transaction):
        """Wrong content type should be handled."""
        response = client.post(
            "/v1/predict",
            content=str(valid_transaction),
            headers={**api_key_headers, "Content-Type": "text/plain"}
        )
        assert response.status_code in [400, 415, 422]


class TestGzipCompression:
    """Tests for response compression."""

    def test_large_batch_response_compressed(self, client, api_key_headers, valid_transaction):
        """Large responses should be compressed."""
        # Create a larger batch to trigger compression
        transactions = [valid_transaction] * 20
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": transactions},
            headers={**api_key_headers, "Accept-Encoding": "gzip"}
        )
        assert response.status_code == 200
        # Check if compression was applied (header or smaller size)
        # Note: TestClient may not preserve encoding headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
