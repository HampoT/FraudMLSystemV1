import os
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)


def evaluate_model(data_dir: str, artifacts_dir: str, reports_dir: str) -> None:
    """Evaluate model on test set and generate reports.

    Args:
        data_dir: Directory containing test split
        artifacts_dir: Directory containing model artifacts
        reports_dir: Directory to save plots/metrics
    """
    print("Loading test data...")
    X_test: pd.DataFrame = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test: np.ndarray = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    print("Loading model...")
    model_path: str = os.path.join(artifacts_dir, 'model.joblib')
    meta_path: str = os.path.join(artifacts_dir, 'model_meta.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta: dict = json.load(f)

    from ..data.features import engineer_features, get_feature_names
    feature_cols: list = get_feature_names()
    X_test_engineered: pd.DataFrame = engineer_features(X_test)
    X_test_final: pd.DataFrame = X_test_engineered[feature_cols]

    y_pred: np.ndarray = model.predict(X_test_final)
    y_prob: np.ndarray = model.predict_proba(X_test_final)[:, 1]

    roc_auc: float = roc_auc_score(y_test, y_prob)
    pr_auc: float = average_precision_score(y_test, y_prob)

    tuned_threshold: float = meta.get("threshold", 0.5)
    y_pred_tuned: np.ndarray = (y_prob >= tuned_threshold).astype(int)

    target_precision: float = meta.get("target_precision")
    test_recall_at_target: float = None
    test_thresh_at_target: float = None

    if target_precision:
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        qualifying_mask = precision[:-1] >= target_precision
        if np.any(qualifying_mask):
            qualifying_indices = np.where(qualifying_mask)[0]
            best_idx = qualifying_indices[np.argmax(recall[qualifying_indices])]
            test_thresh_at_target = float(thresholds[best_idx])
            test_recall_at_target = recall[best_idx]

    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC:  {pr_auc:.4f}")
    if target_precision:
        print(f"Test Recall @ Precision>={target_precision}: {test_recall_at_target}")
        print(f"Test Threshold for target: {test_thresh_at_target}")

    print("\nClassification Report (Test) using tuned threshold:")
    print(classification_report(y_test, y_pred_tuned, target_names=['Legit', 'Fraud']))

    os.makedirs(reports_dir, exist_ok=True)

    metrics: dict = {
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred_tuned).tolist(),
        "test_recall_at_target_precision": test_recall_at_target,
        "test_threshold_at_target_precision": test_thresh_at_target,
        "model_type": meta.get("model_type"),
        "model_version": meta.get("model_version")
    }
    metrics_path: str = os.path.join(reports_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, marker='.', label=f'PR Curve (AUC={pr_auc:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    pr_curve_path: str = os.path.join(reports_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR Curve saved to {pr_curve_path}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, marker='.', label=f'ROC Curve (AUC={roc_auc:.2f})')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_curve_path: str = os.path.join(reports_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC Curve saved to {roc_curve_path}")

    cm = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path: str = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")

    if "feature_importances" in meta:
        importances: dict = meta["feature_importances"]
        sorted_importances: dict = dict(sorted(importances.items(),
                                               key=lambda x: x[1], reverse=True))
        plt.figure(figsize=(10, 8))
        plt.barh(list(sorted_importances.keys()),
                 list(sorted_importances.values()))
        plt.xlabel('Importance', fontsize=12)
        plt.title('Feature Importances', fontsize=14)
        plt.tight_layout()
        importance_path: str = os.path.join(reports_dir, 'feature_importances.png')
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature Importances saved to {importance_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--data-dir", default="data", help="Directory containing test split")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing model artifacts")
    parser.add_argument("--reports-dir", default="reports", help="Directory to save plots/metrics")

    args = parser.parse_args()

    evaluate_model(args.data_dir, args.artifacts_dir, args.reports_dir)
