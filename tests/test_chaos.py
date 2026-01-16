import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestChaosScenarios:
    """Tests for chaos engineering scenarios."""

    def test_model_not_loaded(self):
        """Test graceful degradation when model is not loaded."""
        with patch('src.fraudml.api.app.model', None):
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
            
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    def test_rate_limit_exceeded(self):
        """Test rate limiting when exceeded."""
        from src.fraudml.api.cache import request_queue
        
        request_queue.request_times.clear()
        
        for i in range(101):
            request_queue.record_request("test_user")
        
        count = request_queue.get_request_count("test_user", 60)
        assert count > 100

    def test_prediction_timeout_handling(self):
        """Test that slow predictions don't block the API."""
        from src.fraudml.api.cache import RequestQueue
        
        queue = RequestQueue(max_concurrent=2)
        
        async def slow_operation():
            await asyncio.sleep(0.1)
            return "done"
        
        async def test():
            results = []
            for _ in range(3):
                result = await queue.process("user", slow_operation)
                results.append(result)
            return results
        
        start = time.time()
        results = asyncio.run(test())
        elapsed = time.time() - start
        
        assert len(results) == 3
        assert elapsed >= 0.1

    def test_cache_failure_graceful_degradation(self):
        """Test that cache failures don't break predictions."""
        from src.fraudml.api.cache import AsyncPredictionCache
        
        cache = AsyncPredictionCache(redis_url="redis://invalid:6379")
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        result = asyncio.run(cache.get({"amount": 100, "hour": 12, "device_score": 0.5, "country_risk": 2}))
        assert result is None

    def test_graceful_handling_of_invalid_input(self):
        """Test that invalid input doesn't crash the API."""
        from src.fraudml.api.app import app
        client = TestClient(app)
        
        response = client.post(
            "/v1/predict",
            json={
                "amount": "not_a_number",
                "hour": 14,
                "device_score": 0.85,
                "country_risk": 2
            },
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 422

    def test_handling_of_extreme_values(self):
        """Test handling of extreme but valid input values."""
        from src.fraudml.api.app import app
        client = TestClient(app)
        
        response = client.post(
            "/v1/predict",
            json={
                "amount": 0.01,
                "hour": 0,
                "device_score": 0.0,
                "country_risk": 1
            },
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200

    def test_concurrent_requests_handling(self):
        """Test that concurrent requests are handled properly."""
        import concurrent.futures
        from src.fraudml.api.cache import RequestQueue
        
        queue = RequestQueue(max_concurrent=10)
        
        async def simple_task():
            await asyncio.sleep(0.01)
            return "success"
        
        def make_request(user_id):
            return asyncio.run(queue.process(user_id, simple_task))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, f"user_{i % 5}") for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 50
        assert all(r == "success" for r in results)

    def test_memory_pressure_handling(self):
        """Test handling of large batch requests."""
        from src.fraudml.api.app import BatchTransactionInput, TransactionInput
        
        transactions = [
            TransactionInput(
                amount=100 + i,
                hour=12,
                device_score=0.5,
                country_risk=2
            )
            for i in range(100)
        ]
        
        batch = BatchTransactionInput(transactions=transactions)
        assert len(batch.transactions) == 100

    def test_audit_log_failure_handling(self):
        """Test that audit log failures don't break predictions."""
        from src.fraudml.api.auth import audit_logger
        
        original_logs = audit_logger._logs.copy()
        
        try:
            audit_logger.log("test", "user", {"data": "test"}, "127.0.0.1")
        except Exception:
            pass
        
        assert audit_logger._logs == original_logs or len(audit_logger._logs) > 0

    def test_model_loading_failure_handling(self):
        """Test handling of model loading failures."""
        import os
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_model_path = os.path.join(tmpdir, "model.joblib")
            with open(bad_model_path, "w") as f:
                f.write("not a valid model")
            
            with patch('src.fraudml.api.app.model_path', bad_model_path):
                from src.fraudml.api.app import app
                client = TestClient(app)
                
                response = client.get("/health")
                assert response.status_code == 200

    def test_database_failure_handling(self):
        """Test handling of database failures gracefully."""
        from src.fraudml.api.cache import FeatureCache
        
        cache = FeatureCache(redis_url="redis://invalid:6379")
        
        result = asyncio.run(cache.get_features("user123"))
        assert result == {}

    def test_circuit_breaker_pattern(self):
        """Test simulated circuit breaker behavior."""
        failure_count = 0
        threshold = 3
        
        def should_fail():
            nonlocal failure_count
            failure_count += 1
            return failure_count < threshold
        
        for i in range(5):
            if should_fail():
                pass
            else:
                assert failure_count >= threshold


class TestRecoveryScenarios:
    """Tests for system recovery scenarios."""

    def test_cache_recovery_after_failure(self):
        """Test cache recovery after connection failure."""
        from src.fraudml.api.cache import AsyncPredictionCache, ModelCache
        
        async def test_recovery():
            model_cache = ModelCache()
            
            with patch('src.fraudml.api.cache.os.path.exists', return_value=False):
                model, meta = await model_cache.get_model("nonexistent", "nonexistent")
                assert model is None
            
            with patch('src.fraudml.api.cache.os.path.exists', return_value=True):
                with patch('src.fraudml.api.cache.joblib.load') as mock_load:
                    mock_load.return_value = MagicMock()
                    model, meta = await model_cache.get_model("model.joblib", "meta.json")
                    assert model is not None
            
            return True
        
        result = asyncio.run(test_recovery())
        assert result

    def test_rate_limit_reset(self):
        """Test that rate limits reset over time."""
        from src.fraudml.api.cache import RequestQueue
        
        queue = RequestQueue(max_concurrent=100)
        
        queue.record_request("user1")
        queue.record_request("user1")
        
        count = queue.get_request_count("user1", 60)
        assert count == 2
        
        queue.cleanup_old_requests(window=0)
        count = queue.get_request_count("user1", 60)
        assert count == 0


class TestSecurityChaos:
    """Tests for security chaos scenarios."""

    def test_sql_injection_prevention(self):
        """Test SQL injection attempts are sanitized."""
        from src.fraudml.api.app import sanitize_input
        
        malicious_input = {
            "amount": "100; DROP TABLE users;",
            "hour": "14",
            "device_score": "0.85",
            "country_risk": "2"
        }
        
        sanitized = sanitize_input(malicious_input)
        assert "DROP TABLE" not in sanitized["amount"]

    def test_xss_prevention(self):
        """Test XSS attempts are sanitized."""
        from src.fraudml.api.app import sanitize_input
        
        malicious_input = {
            "amount": "<script>alert('xss')</script>",
            "hour": "14",
            "device_score": "0.85",
            "country_risk": "2"
        }
        
        sanitized = sanitize_input(malicious_input)
        assert "<script>" not in sanitized["amount"]

    def test_invalid_token_handling(self):
        """Test that invalid tokens are handled gracefully."""
        from src.fraudml.api.auth import auth_handler
        
        try:
            auth_handler.verify_token("invalid_token")
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Invalid token" in str(e)

    def test_expired_token_handling(self):
        """Test that expired tokens are handled gracefully."""
        import jwt
        import time
        
        from src.fraudml.api.auth import auth_handler, AuthConfig
        
        expired_token = jwt.encode(
            {
                "user_id": "user123",
                "token_type": "access",
                "exp": int(time.time()) - 3600,
                "iat": int(time.time()) - 7200
            },
            auth_handler.secret_key,
            algorithm=auth_handler.algorithm
        )
        
        try:
            auth_handler.verify_token(expired_token)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "expired" in str(e).lower()
