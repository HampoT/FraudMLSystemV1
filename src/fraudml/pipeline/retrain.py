import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model_type: str
    roc_auc: float
    pr_auc: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    threshold: float
    achieved_precision: float
    achieved_recall: float
    training_time_seconds: float
    timestamp: str
    commit_hash: str


class ModelTrainer:
    """Automated model training with comparison."""

    def __init__(self, data_dir: str = "data", artifacts_dir: str = "artifacts"):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.training_results: List[TrainingResult] = []

    def train_single_model(self, model_type: str, seed: int = 42) -> TrainingResult:
        """Train a single model and return results."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
        from ..data.features import engineer_features, get_feature_names
        import pandas as pd
        import numpy as np
        import joblib
        
        start_time = time.time()
        
        logger.info(f"Training {model_type}...")
        
        X_train = pd.read_csv(os.path.join(self.data_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(self.data_dir, 'y_train.csv')).values.ravel()
        X_val = pd.read_csv(os.path.join(self.data_dir, 'X_val.csv'))
        y_val = pd.read_csv(os.path.join(self.data_dir, 'y_val.csv')).values.ravel()
        
        feature_cols = get_feature_names()
        X_train_eng = engineer_features(X_train)
        X_val_eng = engineer_features(X_val)
        
        X_train_final = X_train_eng[feature_cols]
        X_val_final = X_val_eng[feature_cols]
        
        model_map = {
            'LogisticRegression': LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=seed
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced',
                random_state=seed, n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
                random_state=seed, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
            )
        }
        
        model = model_map.get(model_type)
        if not model:
            raise ValueError(f"Unknown model: {model_type}")
        
        model.fit(X_train_final, y_train)
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(model, X_train_final, y_train, cv=cv_strategy, scoring='roc_auc')
        
        y_prob = model.predict_proba(X_val_final)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        
        precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
        qualifying_mask = precision[:-1] >= 0.95
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
        
        training_time = time.time() - start_time
        
        result = TrainingResult(
            model_type=model_type,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            cv_roc_auc_mean=cv_scores.mean(),
            cv_roc_auc_std=cv_scores.std(),
            threshold=best_threshold,
            achieved_precision=achieved_precision,
            achieved_recall=achieved_recall,
            training_time_seconds=training_time,
            timestamp=datetime.utcnow().isoformat(),
            commit_hash=self._get_commit_hash()
        )
        
        self.training_results.append(result)
        
        model_artifacts_dir = os.path.join(self.artifacts_dir, model_type.lower())
        os.makedirs(model_artifacts_dir, exist_ok=True)
        
        model_path = os.path.join(model_artifacts_dir, 'model.joblib')
        joblib.dump(model, model_path)
        
        meta = {
            "model_version": datetime.utcnow().isoformat(),
            "model_type": model_type,
            "features": feature_cols,
            "threshold": best_threshold,
            "target_precision": 0.95,
            "metrics_val": {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "cv_roc_auc_mean": cv_scores.mean(),
                "cv_roc_auc_std": cv_scores.std()
            },
            "threshold_metrics": {
                "achieved_precision": achieved_precision,
                "achieved_recall": achieved_recall
            },
            "seed": seed,
            "training_time_seconds": training_time,
            "commit_hash": result.commit_hash
        }
        
        if hasattr(model, 'feature_importances_'):
            meta["feature_importances"] = dict(zip(feature_cols, model.feature_importances_.tolist()))
        
        meta_path = os.path.join(model_artifacts_dir, 'model_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Trained {model_type}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        
        return result

    def _get_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        except:
            return "unknown"

    def train_all_models(self, models: List[str] = None) -> Dict:
        """Train all models and return comparison."""
        if models is None:
            models = ['LogisticRegression', 'RandomForest', 'XGBoost']
        
        results = {}
        for model_type in models:
            try:
                result = self.train_single_model(model_type)
                results[model_type] = {
                    "roc_auc": result.roc_auc,
                    "pr_auc": result.pr_auc,
                    "cv_roc_auc": result.cv_roc_auc_mean,
                    "cv_std": result.cv_roc_auc_std,
                    "threshold": result.threshold,
                    "precision": result.achieved_precision,
                    "recall": result.achieved_recall,
                    "training_time": result.training_time_seconds
                }
            except Exception as e:
                logger.error(f"Training {model_type} failed: {e}")
                results[model_type] = {"error": str(e)}
        
        best_model = max(
            [(k, v.get("roc_auc", 0)) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1]
        )[0] if results else None
        
        return {
            "results": results,
            "best_model": best_model,
            "timestamp": datetime.utcnow().isoformat()
        }

    def compare_models(self) -> Dict:
        """Compare trained models."""
        if not self.training_results:
            return {"error": "No training results available"}
        
        comparison = {
            "models": {},
            "best_by_roc_auc": None,
            "best_by_pr_auc": None,
            "best_by_recall": None
        }
        
        for result in self.training_results:
            comparison["models"][result.model_type] = {
                "roc_auc": result.roc_auc,
                "pr_auc": result.pr_auc,
                "cv_roc_auc": result.cv_roc_auc_mean,
                "threshold": result.threshold,
                "achieved_recall": result.achieved_recall,
                "training_time": result.training_time_seconds
            }
        
        if comparison["models"]:
            comparison["best_by_roc_auc"] = max(
                comparison["models"].items(),
                key=lambda x: x[1]["roc_auc"]
            )[0]
            comparison["best_by_pr_auc"] = max(
                comparison["models"].items(),
                key=lambda x: x[1]["pr_auc"]
            )[0]
        
        return comparison

    def save_results(self, output_path: str = "reports/training_results.json"):
        """Save training results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        comparison = self.compare_models()
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Training results saved to {output_path}")
        return comparison


def trigger_retraining(data_changed: bool = False, metric_degraded: bool = False) -> bool:
    """Check if retraining should be triggered."""
    if data_changed:
        logger.info("Data changes detected - triggering retraining")
        return True
    if metric_degraded:
        logger.info("Model performance degradation detected - triggering retraining")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Automated model training")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--model", default=None, help="Single model to train")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--compare", action="store_true", help="Compare models")
    parser.add_argument("--output", default="reports/training_results.json", help="Output path")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.data_dir, args.artifacts_dir)
    
    if args.all:
        results = trainer.train_all_models()
        print(f"Trained all models. Best model: {results.get('best_model')}")
    elif args.model:
        result = trainer.train_single_model(args.model)
        print(f"Trained {args.model}: ROC-AUC={result.roc_auc:.4f}")
    elif args.compare:
        comparison = trainer.compare_models()
        print(json.dumps(comparison, indent=2))
    else:
        results = trainer.train_all_models()
        trainer.save_results(args.output)
        print(f"Results saved to {args.output}")
        print(f"Best model: {results.get('best_model')}")


if __name__ == "__main__":
    main()
