import os
import argparse
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from datetime import datetime

def train_model(data_dir, artifacts_dir, seed=42):
    """
    Trains a logistic regression baseline model and saves artifacts.
    """
    # Load data
    print("Loading data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv')).values.ravel()
    
    # Train model
    # balanced class weight is critical for imbalanced fraud data
    print("Training LogisticRegression...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    
    # Predict probabilities (needed for threshold tuning)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # --- Threshold Tuning ---
    target_precision = 0.95
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    
    # Find thresholds where precision >= target
    # precision matches thresholds length + 1 (last is 1.0), so we slice precision[:-1]
    qualifying_mask = precision[:-1] >= target_precision
    
    if np.any(qualifying_mask):
        # Among qualifying thresholds, pick the one with highest recall
        # (recall is parallel to precision[:-1])
        qualifying_indices = np.where(qualifying_mask)[0]
        best_idx = qualifying_indices[np.argmax(recall[qualifying_indices])]
        
        best_threshold = float(thresholds[best_idx])
        achieved_precision = precision[best_idx]
        achieved_recall = recall[best_idx]
        
        print(f"\nThreshold Tuning:")
        print(f"  Target Precision: {target_precision}")
        print(f"  Chosen Threshold: {best_threshold:.4f}")
        print(f"  Achieved Precision: {achieved_precision:.4f}")
        print(f"  Achieved Recall:    {achieved_recall:.4f}")
    else:
        best_threshold = 0.5
        print(f"\nThreshold Tuning:")
        print(f"  Target Precision: {target_precision}")
        print(f"  No threshold met target. Fallback to 0.5.")

    # Generate predictions using the chosen threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    
    print(f"\nValidation Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print("\nClassification Report (Val):")
    print(classification_report(y_val, y_pred))
    
    # Save artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Save model
    model_path = os.path.join(artifacts_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # 2. Save metadata
    meta = {
        "model_version": datetime.now().isoformat(),
        "algorithm": "LogisticRegression",
        "features": list(X_train.columns),
        "threshold": best_threshold,
        "target_precision": target_precision,
        "metrics_val": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        },
        "seed": seed
    }
    meta_path = os.path.join(artifacts_dir, 'model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data-dir", default="data", help="Directory containing train/val splits")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to save model artifacts")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.artifacts_dir)
