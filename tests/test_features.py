import pytest
import pandas as pd
import numpy as np
from src.fraudml.data.features import engineer_features, get_feature_names, scale_features


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            'amount': [100, 500, 2000, 5000, 100],
            'hour': [10, 14, 3, 22, 12],
            'device_score': [0.9, 0.8, 0.2, 0.1, 0.95],
            'country_risk': [1, 2, 5, 4, 1]
        })

    def test_engineer_features_basic(self, sample_data):
        """Test that feature engineering adds new columns."""
        result = engineer_features(sample_data)

        assert 'amount_log' in result.columns
        assert 'amount_zscore' in result.columns
        assert 'is_night' in result.columns
        assert 'is_evening' in result.columns
        assert 'risk_score' in result.columns

    def test_engineer_features_shape(self, sample_data):
        """Test that feature engineering preserves row count."""
        original_shape = sample_data.shape
        result = engineer_features(sample_data)

        assert result.shape[0] == original_shape[0]

    def test_amount_log_transform(self):
        """Test that log transform is applied correctly."""
        df = pd.DataFrame({'amount': [100], 'hour': [12], 'device_score': [0.5], 'country_risk': [2]})
        result = engineer_features(df)

        expected_log = np.log1p(100)
        assert abs(result['amount_log'].values[0] - expected_log) < 0.0001

    def test_night_flag(self):
        """Test that night hours are correctly flagged."""
        df_night = pd.DataFrame({
            'amount': [100], 'hour': [3], 'device_score': [0.5], 'country_risk': [2]
        })
        df_day = pd.DataFrame({
            'amount': [100], 'hour': [14], 'device_score': [0.5], 'country_risk': [2]
        })

        result_night = engineer_features(df_night)
        result_day = engineer_features(df_day)

        assert result_night['is_night'].values[0] == 1
        assert result_day['is_night'].values[0] == 0

    def test_high_amount_flag(self):
        """Test that high amounts are correctly flagged."""
        low_amount = pd.DataFrame({
            'amount': [50], 'hour': [12], 'device_score': [0.5], 'country_risk': [2]
        })
        high_amount = pd.DataFrame({
            'amount': [5000], 'hour': [12], 'device_score': [0.5], 'country_risk': [2]
        })

        result_low = engineer_features(low_amount)
        result_high = engineer_features(high_amount)

        assert result_low['high_amount'].values[0] == 0
        assert result_high['high_amount'].values[0] == 1

    def test_low_device_flag(self):
        """Test that low device scores are correctly flagged."""
        high_device = pd.DataFrame({
            'amount': [100], 'hour': [12], 'device_score': [0.9], 'country_risk': [2]
        })
        low_device = pd.DataFrame({
            'amount': [100], 'hour': [12], 'device_score': [0.1], 'country_risk': [2]
        })

        result_high = engineer_features(high_device)
        result_low = engineer_features(low_device)

        assert result_high['low_device'].values[0] == 0
        assert result_low['low_device'].values[0] == 1

    def test_risk_country_flag(self):
        """Test that high-risk countries are correctly flagged."""
        low_risk = pd.DataFrame({
            'amount': [100], 'hour': [12], 'device_score': [0.5], 'country_risk': [2]
        })
        high_risk = pd.DataFrame({
            'amount': [100], 'hour': [12], 'device_score': [0.5], 'country_risk': [5]
        })

        result_low = engineer_features(low_risk)
        result_high = engineer_features(high_risk)

        assert result_low['high_risk_country'].values[0] == 0
        assert result_high['high_risk_country'].values[0] == 1

    def test_cyclical_hour_encoding(self):
        """Test that hour is encoded cyclically."""
        df = pd.DataFrame({
            'amount': [100], 'hour': [0], 'device_score': [0.5], 'country_risk': [2]
        })
        result = engineer_features(df)

        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns

        assert abs(result['hour_sin'].values[0]) <= 1.0
        assert abs(result['hour_cos'].values[0]) <= 1.0

    def test_interaction_features(self):
        """Test that interaction features are created."""
        df = pd.DataFrame({
            'amount': [1000], 'hour': [12], 'device_score': [0.5], 'country_risk': [3]
        })
        result = engineer_features(df)

        assert 'amount_device_interaction' in result.columns
        assert 'amount_country_interaction' in result.columns

        expected_interaction = 1000 * (1 - 0.5)
        assert abs(result['amount_device_interaction'].values[0] - expected_interaction) < 0.0001

    def test_get_feature_names(self):
        """Test that feature names are returned."""
        feature_names = get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'amount' in feature_names
        assert 'amount_log' in feature_names
        assert 'risk_score' in feature_names

    def test_scale_features(self):
        """Test that feature scaling works correctly."""
        train_df = pd.DataFrame({
            'amount': [100, 200, 300],
            'hour': [10, 12, 14],
            'device_score': [0.5, 0.6, 0.7],
            'country_risk': [1, 2, 3],
            'is_fraud': [0, 0, 1]
        })
        val_df = pd.DataFrame({
            'amount': [150],
            'hour': [11],
            'device_score': [0.55],
            'country_risk': [2],
            'is_fraud': [0]
        })

        X_train, X_val, _, scaler = scale_features(train_df, val_df)

        assert X_train is not None
        assert X_val is not None
        assert scaler is not None

        train_mean = X_train['amount'].mean()
        assert abs(train_mean) < 0.0001

    def test_engineer_features_preserves_original(self, sample_data):
        """Test that original features are preserved."""
        result = engineer_features(sample_data)

        assert 'amount' in result.columns
        assert 'hour' in result.columns
        assert 'device_score' in result.columns
        assert 'country_risk' in result.columns

    def test_edge_case_zero_amount(self):
        """Test handling of edge case with zero amount."""
        df = pd.DataFrame({
            'amount': [0], 'hour': [12], 'device_score': [0.5], 'country_risk': [2]
        })
        result = engineer_features(df)

        assert result['amount_log'].values[0] == 0

    def test_edge_case_boundary_hours(self):
        """Test boundary hours (0 and 23)."""
        df_0 = pd.DataFrame({
            'amount': [100], 'hour': [0], 'device_score': [0.5], 'country_risk': [2]
        })
        df_23 = pd.DataFrame({
            'amount': [100], 'hour': [23], 'device_score': [0.5], 'country_risk': [2]
        })

        result_0 = engineer_features(df_0)
        result_23 = engineer_features(df_23)

        assert result_0['is_night'].values[0] == 1
        assert result_23['is_night'].values[0] == 1
        assert result_23['is_evening'].values[0] == 1


class TestFeatureEngineerForInference:
    """Tests for inference-specific feature engineering."""

    def test_single_transaction_dict(self):
        """Test feature engineering from dictionary."""
        data = {
            'amount': 1500.0,
            'hour': 14,
            'device_score': 0.8,
            'country_risk': 3
        }

        result = engineer_features_for_inference(data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 1
        assert 'amount_log' in result.columns

    def test_multiple_transactions(self):
        """Test feature engineering for multiple transactions."""
        data = [
            {'amount': 100, 'hour': 10, 'device_score': 0.9, 'country_risk': 1},
            {'amount': 5000, 'hour': 3, 'device_score': 0.1, 'country_risk': 5}
        ]

        df = pd.DataFrame(data)
        result = engineer_features(df)

        assert result.shape[0] == 2
