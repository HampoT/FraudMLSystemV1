import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..data.features import engineer_features, get_feature_names


def batch_predict_from_file(input_path: str, output_path: str = None):
    """Run batch predictions from a CSV file.

    Args:
        input_path: Path to input CSV with transactions
        output_path: Optional path to save predictions
    """
    import joblib
    import json

    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    meta_path = os.getenv("META_PATH", "artifacts/model_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} transactions from {input_path}")

    feature_cols = get_feature_names()
    engineered = engineer_features(df)
    features = engineered[feature_cols]

    threshold = meta.get("threshold", 0.5)
    probs = model.predict_proba(features)[:, 1]
    labels = (probs >= threshold).astype(int)

    results = pd.DataFrame({
        'fraud_probability': probs,
        'fraud_label': labels,
        'threshold_used': threshold
    })

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    print(f"\nPrediction Summary:")
    print(f"  Total transactions: {len(df)}")
    print(f"  Fraud predictions: {labels.sum()}")
    print(f"  Fraud rate: {labels.mean():.4f}")
    print(f"  Mean fraud probability: {probs.mean():.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run batch predictions")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default="reports/batch_predictions.csv",
                        help="Output CSV file path")

    args = parser.parse_args()

    batch_predict_from_file(args.input, args.output)
