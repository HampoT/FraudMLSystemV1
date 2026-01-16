import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to fraud detection dataset.

    Args:
        df: Raw DataFrame with columns: amount, hour, device_score, country_risk

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    df['amount_log'] = np.log1p(df['amount'])

    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)
    df['is_weekend'] = 0

    df['high_amount'] = (df['amount'] > df['amount'].quantile(0.75)).astype(int)
    df['very_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    df['low_device'] = (df['device_score'] < 0.3).astype(int)
    df['medium_device'] = ((df['device_score'] >= 0.3) & (df['device_score'] < 0.7)).astype(int)

    df['high_risk_country'] = (df['country_risk'] >= 4).astype(int)

    df['risk_score'] = (
        df['high_amount'] * 0.3 +
        df['low_device'] * 0.3 +
        df['high_risk_country'] * 0.2 +
        df['is_night'] * 0.2
    )

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['amount_device_interaction'] = df['amount'] * (1 - df['device_score'])
    df['amount_country_interaction'] = df['amount'] * df['country_risk'] / 5

    return df


def engineer_features_for_inference(data: dict) -> pd.DataFrame:
    """Apply feature engineering to a single transaction for inference.

    Args:
        data: Dictionary with transaction features

    Returns:
        DataFrame with engineered features
    """
    df = pd.DataFrame([data])
    return engineer_features(df)


def get_feature_names() -> list:
    """Return list of all feature names used in model.

    Returns:
        List of feature column names
    """
    return [
        'amount', 'hour', 'device_score', 'country_risk',
        'amount_log', 'amount_zscore',
        'is_night', 'is_evening', 'is_weekend',
        'high_amount', 'very_high_amount',
        'low_device', 'medium_device',
        'high_risk_country',
        'risk_score',
        'hour_sin', 'hour_cos',
        'amount_device_interaction', 'amount_country_interaction'
    ]


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                   test_df: pd.DataFrame = None) -> tuple:
    """Scale features using StandardScaler fitted on training data.

    Args:
        train_df: Training DataFrame
        val_df: Optional validation DataFrame
        test_df: Optional test DataFrame

    Returns:
        Tuple of scaled DataFrames
    """
    scaler = StandardScaler()
    feature_cols = [c for c in train_df.columns if c not in ['is_fraud']]

    X_train = train_df[feature_cols].copy()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )

    if val_df is not None:
        X_val = val_df[feature_cols].copy()
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=feature_cols,
            index=X_val.index
        )
    else:
        X_val_scaled = None

    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index
        )
    else:
        X_test_scaled = None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def add_transaction_features(df: pd.DataFrame,
                             transaction_counts: dict = None,
                             user_history: dict = None) -> pd.DataFrame:
    """Add transaction-level features if available.

    Args:
        df: DataFrame with transactions
        transaction_counts: Dict mapping user_id to transaction count
        user_history: Dict mapping user_id to historical data

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    if transaction_counts is not None:
        df['tx_count_last_hour'] = df['user_id'].map(transaction_counts).fillna(0)

    if user_history is not None:
        avg_amounts = {k: v.get('avg_amount', 100) for k, v in user_history.items()}
        df['user_avg_amount'] = df['user_id'].map(avg_amounts).fillna(df['amount'].mean())
        df['amount_vs_avg'] = df['amount'] / (df['user_avg_amount'] + 1)

        fraud_rates = {k: v.get('fraud_rate', 0.01) for k, v in user_history.items()}
        df['user_fraud_rate'] = df['user_id'].map(fraud_rates).fillna(0.01)

    return df
