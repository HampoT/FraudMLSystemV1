import os
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, auc, roc_curve, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

def evaluate_model(data_dir, artifacts_dir, reports_dir):
    """
    Evaluates model on test set and generates plots/metrics.
    """
    # Load data
    print("Loading test data...")
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    # Load model and meta
    print("Loading model...")
    model_path = os.path.join(artifacts_dir, 'model.joblib')
    meta_path = os.path.join(artifacts_dir, 'model_meta.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # --- Threshold Metrics on Test ---
    target_precision = meta.get("target_precision")
    tuned_threshold = meta.get("threshold", 0.5)
    
    test_recall_at_target = None
    test_thresh_at_target = None
    
    if target_precision:
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        # Find best recall at target precision on TEST set (to see if it generalizes)
        qualifying_mask = precision[:-1] >= target_precision
        if np.any(qualifying_mask):
             qualifying_indices = np.where(qualifying_mask)[0]
             best_idx = qualifying_indices[np.argmax(recall[qualifying_indices])]
             test_thresh_at_target = float(thresholds[best_idx])
             test_recall_at_target = recall[best_idx]
    
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC:  {pr_auc:.4f}")
    if target_precision:
        print(f"Test Recall @ Precision>={target_precision}: {test_recall_at_target if test_recall_at_target else 'N/A'}")
        print(f"Test Threshold for target (generalization check): {test_thresh_at_target if test_thresh_at_target else 'N/A'}")
    
    print("\nClassification Report (Test) using tuned threshold ({:.4f}):".format(tuned_threshold))
    # Regenerate predictions using tuned threshold
    y_pred_tuned = (y_prob >= tuned_threshold).astype(int)
    print(classification_report(y_test, y_pred_tuned))
    
    # Save Metrics JSON
    os.makedirs(reports_dir, exist_ok=True)
    metrics = {
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred_tuned).tolist(),
        "test_recall_at_target_precision": test_recall_at_target,
        "test_threshold_at_target_precision": test_thresh_at_target
    }
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Plot PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'PR Curve (AUC={pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    pr_curve_path = os.path.join(reports_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"PR Curve saved to {pr_curve_path}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Legit', 'Fraud'], rotation=45)
    plt.yticks(tick_marks, ['Legit', 'Fraud'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--data-dir", default="data", help="Directory containing test split")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing model artifacts")
    parser.add_argument("--reports-dir", default="reports", help="Directory to save plots/metrics")
    
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.artifacts_dir, args.reports_dir)
