import os
import argparse
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve)
from datetime import datetime
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from ..data.features import engineer_features, get_feature_names


def train_logistic_regression(X_train, y_train, X_val, y_val, seed=42):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=seed,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, X_val, y_val, seed=42):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val, y_val, seed=42):
    """Train XGBoost model."""
    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def cross_validate_model(model, X_train, y_train, cv=5):
    """Perform cross-validation and return scores."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
    return {
        'mean_auc': scores.mean(),
        'std_auc': scores.std(),
        'scores': scores.tolist()
    }


def tune_threshold(y_val, y_prob, target_precision=0.95):
    """Tune decision threshold for target precision."""
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    qualifying_mask = precision[:-1] >= target_precision

    if np.any(qualifying_mask):
        qualifying_indices = np.where(qualifying_mask)[0]
        best_idx = qualifying_indices[np.argmax(recall[qualifying_indices])]
        best_threshold = float(thresholds[best_idx])
        achieved_precision = float(precision[best_idx])
        achieved_recall = float(recall[best_idx])
    else:
        best_threshold = 0.5
        achieved_precision = float(precision[np.argmax(np.abs(thresholds - 0.5))])
        achieved_recall = float(recall[np.argmax(np.abs(thresholds - 0.5))])

    return best_threshold, achieved_precision, achieved_recall


def train_model(data_dir, artifacts_dir, model_type='LogisticRegression', seed=42):
    """Train a fraud detection model with cross-validation.

    Args:
        data_dir: Directory containing train/val splits
        artifacts_dir: Directory to save model artifacts
        model_type: Model to train ('LogisticRegression', 'RandomForest', 'XGBoost')
        seed: Random seed
    """
    print(f"Loading data from {data_dir}...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv')).values.ravel()

    print(f"Applying feature engineering...")
    feature_cols = get_feature_names()
    X_train_engineered = engineer_features(X_train)
    X_val_engineered = engineer_features(X_val)

    X_train_final = X_train_engineered[feature_cols]
    X_val_final = X_val_engineered[feature_cols]

    print(f"Training {model_type}...")
    model_map = {
        'LogisticRegression': train_logistic_regression,
        'RandomForest': train_random_forest,
        'XGBoost': train_xgboost
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model: {model_type}. Choose from: {list(model_map.keys())}")

    model = model_map[model_type](X_train_final, y_train, X_val_final, y_val, seed)

    print(f"Performing cross-validation...")
    cv_scores = cross_validate_model(model, X_train_final, y_train)

    y_prob = model.predict_proba(X_val_final)[:, 1]
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)

    print(f"\nCross-Validation Results:")
    print(f"  ROC-AUC: {cv_scores['mean_auc']:.4f} (+/- {cv_scores['std_auc']:.4f})")

    print(f"\nValidation Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")

    best_threshold, achieved_precision, achieved_recall = tune_threshold(y_val, y_prob)
    print(f"\nThreshold Tuning:")
    print(f"  Target Precision: 0.95")
    print(f"  Chosen Threshold: {best_threshold:.4f}")
    print(f"  Achieved Precision: {achieved_precision:.4f}")
    print(f"  Achieved Recall: {achieved_recall:.4f}")

    y_pred = (y_prob >= best_threshold).astype(int)
    print("\nClassification Report (Val):")
    print(classification_report(y_val, y_pred, target_names=['Legit', 'Fraud']))

    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    meta = {
        "model_version": datetime.now().isoformat(),
        "model_type": model_type,
        "features": feature_cols,
        "threshold": best_threshold,
        "target_precision": 0.95,
        "metrics_val": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "cv_roc_auc_mean": cv_scores['mean_auc'],
            "cv_roc_auc_std": cv_scores['std_auc']
        },
        "threshold_metrics": {
            "achieved_precision": achieved_precision,
            "achieved_recall": achieved_recall
        },
        "seed": seed,
        "training_samples": len(y_train),
        "feature_engineering": True
    }

    if hasattr(model, 'feature_importances_'):
        importance_dict = dict(zip(feature_cols, model.feature_importances_.tolist()))
        meta["feature_importances"] = importance_dict

    meta_path = os.path.join(artifacts_dir, 'model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return model, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data-dir", default="data", help="Directory containing train/val splits")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to save model artifacts")
    parser.add_argument("--model", default="LogisticRegression",
                        choices=['LogisticRegression', 'RandomForest', 'XGBoost'],
                        help="Model type to train")

    args = parser.parse_args()

    train_model(args.data_dir, args.artifacts_dir, args.model)
