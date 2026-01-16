import pytest
import pandas as pd
import numpy as np


class TestDataDownload:
    """Tests for data download/generation module."""

    def test_generate_toy_data_shape(self):
        """Test that toy data has correct shape."""
        from src.fraudml.data.download import generate_toy_data

        n_samples = 1000
        df = generate_toy_data("/tmp/test_data.csv", n_samples=n_samples, seed=42)

        assert len(df) == n_samples
        assert list(df.columns) == ['amount', 'hour', 'device_score', 'country_risk', 'is_fraud']

    def test_generate_toy_data_fraud_rate(self):
        """Test that fraud rate is within expected range."""
        from src.fraudml.data.download import generate_toy_data

        df = generate_toy_data("/tmp/test_data.csv", n_samples=10000, seed=42)

        fraud_rate = df['is_fraud'].mean()
        assert 0.005 < fraud_rate < 0.05

    def test_generate_toy_data_deterministic(self):
        """Test that same seed produces same data."""
        from src.fraudml.data.download import generate_toy_data

        df1 = generate_toy_data("/tmp/test1.csv", n_samples=1000, seed=123)
        df2 = generate_toy_data("/tmp/test2.csv", n_samples=1000, seed=123)

        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_toy_data_column_ranges(self):
        """Test that generated data is within expected ranges."""
        from src.fraudml.data.download import generate_toy_data

        df = generate_toy_data("/tmp/test_data.csv", n_samples=1000, seed=42)

        assert df['amount'].min() >= 0
        assert df['hour'].min() >= 0
        assert df['hour'].max() <= 23
        assert df['device_score'].min() >= 0
        assert df['device_score'].max() <= 1
        assert df['country_risk'].min() >= 1
        assert df['country_risk'].max() <= 5

    def test_generate_toy_data_different_seeds(self):
        """Test that different seeds produce different data."""
        from src.fraudml.data.download import generate_toy_data

        df1 = generate_toy_data("/tmp/test1.csv", n_samples=1000, seed=1)
        df2 = generate_toy_data("/tmp/test2.csv", n_samples=1000, seed=2)

        assert not df1.equals(df2)


class TestDataPreprocessing:
    """Tests for data preprocessing module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for preprocessing tests."""
        return pd.DataFrame({
            'amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'hour': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'device_score': [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.95],
            'country_risk': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'is_fraud': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        })

    def test_preprocess_splits_data(self, sample_data):
        """Test that preprocessing splits data correctly."""
        from src.fraudml.data.preprocess import preprocess_data

        preprocess_data("/tmp/raw.csv", "/tmp/splits", seed=42)

        X_train = pd.read_csv("/tmp/splits/X_train.csv")
        X_val = pd.read_csv("/tmp/splits/X_val.csv")
        X_test = pd.read_csv("/tmp/splits/X_test.csv")

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(sample_data)

    def test_preprocess_stratification(self, sample_data):
        """Test that preprocessing stratifies by fraud label."""
        from src.fraudml.data.preprocess import preprocess_data

        preprocess_data("/tmp/raw.csv", "/tmp/splits", seed=42)

        y_train = pd.read_csv("/tmp/splits/y_train.csv")
        y_val = pd.read_csv("/tmp/splits/y_val.csv")
        y_test = pd.read_csv("/tmp/splits/y_test.csv")

        original_fraud_rate = sample_data['is_fraud'].mean()

        assert abs(y_train['is_fraud'].mean() - original_fraud_rate) < 0.1
        assert abs(y_val['is_fraud'].mean() - original_fraud_rate) < 0.1
        assert abs(y_test['is_fraud'].mean() - original_fraud_rate) < 0.1

    def test_preprocess_removes_target(self, sample_data):
        """Test that preprocessing removes target from features."""
        from src.fraudml.data.preprocess import preprocess_data

        preprocess_data("/tmp/raw.csv", "/tmp/splits", seed=42)

        X_train = pd.read_csv("/tmp/splits/X_train.csv")

        assert 'is_fraud' not in X_train.columns

    def test_preprocess_split_ratios(self):
        """Test that split ratios are correct (60/20/20)."""
        from src.fraudml.data.preprocess import preprocess_data

        n_samples = 1000
        raw_df = pd.DataFrame({
            'amount': np.random.exponential(100, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'device_score': np.random.beta(5, 2, n_samples),
            'country_risk': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'is_fraud': np.random.binomial(1, 0.01, n_samples)
        })
        raw_df.to_csv("/tmp/raw.csv", index=False)

        preprocess_data("/tmp/raw.csv", "/tmp/splits", seed=42)

        X_train = pd.read_csv("/tmp/splits/X_train.csv")
        X_val = pd.read_csv("/tmp/splits/X_val.csv")
        X_test = pd.read_csv("/tmp/splits/X_test.csv")

        assert len(X_train) == 600
        assert len(X_val) == 200
        assert len(X_test) == 200

    def test_preprocess_creates_directories(self, tmp_path):
        """Test that preprocessing creates output directories."""
        from src.fraudml.data.preprocess import preprocess_data
        import os

        raw_file = tmp_path / "raw.csv"
        output_dir = tmp_path / "splits"

        raw_df = pd.DataFrame({
            'amount': [100, 200, 300],
            'hour': [10, 11, 12],
            'device_score': [0.5, 0.6, 0.7],
            'country_risk': [1, 2, 3],
            'is_fraud': [0, 0, 1]
        })
        raw_df.to_csv(raw_file, index=False)

        preprocess_data(str(raw_file), str(output_dir), seed=42)

        assert os.path.exists(output_dir / "X_train.csv")
        assert os.path.exists(output_dir / "X_val.csv")
        assert os.path.exists(output_dir / "X_test.csv")


class TestModelTraining:
    """Tests for model training module."""

    @pytest.fixture
    def training_data(self, tmp_path):
        """Create training data for model tests."""
        X_train = pd.DataFrame({
            'amount': [100, 200, 300, 400, 500, 600],
            'hour': [10, 11, 12, 13, 14, 15],
            'device_score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            'country_risk': [1, 2, 3, 4, 5, 1]
        })
        y_train = pd.DataFrame({'is_fraud': [0, 0, 0, 0, 1, 1]})

        X_val = pd.DataFrame({
            'amount': [150, 250],
            'hour': [10, 12],
            'device_score': [0.85, 0.65],
            'country_risk': [2, 3]
        })
        y_val = pd.DataFrame({'is_fraud': [0, 1]})

        X_train.to_csv(tmp_path / "X_train.csv", index=False)
        y_train.to_csv(tmp_path / "y_train.csv", index=False)
        X_val.to_csv(tmp_path / "X_val.csv", index=False)
        y_val.to_csv(tmp_path / "y_val.csv", index=False)

        return tmp_path

    def test_train_logistic_regression(self, training_data):
        """Test that Logistic Regression training works."""
        from src.fraudml.models.train import train_logistic_regression
        import os

        X_train = pd.read_csv(training_data / "X_train.csv")
        y_train = pd.read_csv(training_data / "y_train.csv").values.ravel()
        X_val = pd.read_csv(training_data / "X_val.csv")
        y_val = pd.read_csv(training_data / "y_val.csv").values.ravel()

        model = train_logistic_regression(X_train, y_train, X_val, y_val, seed=42)

        assert model is not None
        assert hasattr(model, 'predict_proba')

    def test_train_random_forest(self, training_data):
        """Test that Random Forest training works."""
        from src.fraudml.models.train import train_random_forest

        X_train = pd.read_csv(training_data / "X_train.csv")
        y_train = pd.read_csv(training_data / "y_train.csv").values.ravel()
        X_val = pd.read_csv(training_data / "X_val.csv")
        y_val = pd.read_csv(training_data / "y_val.csv").values.ravel()

        model = train_random_forest(X_train, y_train, X_val, y_val, seed=42)

        assert model is not None
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'feature_importances_')

    def test_train_xgboost(self, training_data):
        """Test that XGBoost training works."""
        from src.fraudml.models.train import train_xgboost

        X_train = pd.read_csv(training_data / "X_train.csv")
        y_train = pd.read_csv(training_data / "y_train.csv").values.ravel()
        X_val = pd.read_csv(training_data / "X_val.csv")
        y_val = pd.read_csv(training_data / "y_val.csv").values.ravel()

        model = train_xgboost(X_train, y_train, X_val, y_val, seed=42)

        assert model is not None
        assert hasattr(model, 'predict_proba')

    def test_threshold_tuning(self):
        """Test that threshold tuning works correctly."""
        from src.fraudml.models.train import tune_threshold

        y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        threshold, precision, recall = tune_threshold(y_val, y_prob, target_precision=0.95)

        assert 0 < threshold < 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    def test_cross_validation(self):
        """Test that cross-validation works."""
        from src.fraudml.models.train import cross_validate_model
        from sklearn.linear_model import LogisticRegression

        X = pd.DataFrame({
            'amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'hour': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        scores = cross_validate_model(model, X, y, cv=3)

        assert 'mean_auc' in scores
        assert 'std_auc' in scores
        assert 'scores' in scores
        assert len(scores['scores']) == 3
        assert 0 <= scores['mean_auc'] <= 1
