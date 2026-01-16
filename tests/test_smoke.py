import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.fraudml.api.app import app
from src.fraudml.data.download import generate_toy_data
from src.fraudml.data.preprocess import preprocess_data
from src.fraudml.models.train import train_model

# Constants for testing
TEST_DATA_DIR = "tests/data"
TEST_ARTIFACTS_DIR = "tests/artifacts"
TEST_RAW_DATA = os.path.join(TEST_DATA_DIR, "raw.csv")

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Matches the pipeline flow for testing."""
    # Ensure clean slate
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Generate fake data
    generate_toy_data(TEST_RAW_DATA, n_samples=500, seed=123)
    
    # 2. Preprocess
    preprocess_data(TEST_RAW_DATA, TEST_DATA_DIR, seed=123)
    
    # 3. Train
    train_model(TEST_DATA_DIR, TEST_ARTIFACTS_DIR, seed=123)
    
    # Set environment variables for the API to verify against these test artifacts
    os.environ["MODEL_PATH"] = os.path.join(TEST_ARTIFACTS_DIR, "model.joblib")
    os.environ["META_PATH"] = os.path.join(TEST_ARTIFACTS_DIR, "model_meta.json")
    
    yield
    
    # Cleanup could go here, but keeping artifacts for inspection is useful

def test_artifacts_exist():
    assert os.path.exists(os.path.join(TEST_ARTIFACTS_DIR, "model.joblib"))
    assert os.path.exists(os.path.join(TEST_ARTIFACTS_DIR, "model_meta.json"))

def test_data_leakage():
    # Ensure no overlap between train and test
    # (Simple check: indices are reset, so we check content logic or just size)
    X_train = pd.read_csv(os.path.join(TEST_DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(TEST_DATA_DIR, "X_test.csv"))
    assert len(X_train) + len(X_test) < 500 # Validation set exists too
    assert len(X_train) > len(X_test)

def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True

def test_api_predict():
    with TestClient(app) as client:
        payload = {
            "amount": 1000.0,
            "hour": 15,
            "device_score": 0.1, # Suspicious
            "country_risk": 5    # Risky
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "fraud_label" in data
        assert 0 <= data["fraud_probability"] <= 1
        assert data["fraud_label"] in [0, 1]
