import os
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime


def log_model_to_mlflow(model, model_type: str, X_train, y_train, meta: dict):
    """Log model to MLflow registry.

    Args:
        model: Trained model
        model_type: Type of model (LogisticRegression, RandomForest, XGBoost)
        X_train: Training features
        y_train: Training labels
        meta: Model metadata dictionary
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("feature_engineering", str(meta.get("feature_engineering", False)))

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("target_precision", meta.get("target_precision", 0.95))
        mlflow.log_param("threshold", meta.get("threshold", 0.5))
        mlflow.log_param("seed", meta.get("seed", 42))

        mlflow.log_metric("roc_auc", meta["metrics_val"]["roc_auc"])
        mlflow.log_metric("pr_auc", meta["metrics_val"]["pr_auc"])
        mlflow.log_metric("cv_roc_auc", meta["metrics_val"]["cv_roc_auc_mean"])
        mlflow.log_metric("cv_roc_auc_std", meta["metrics_val"]["cv_roc_auc_std"])

        mlflow.log_metric("achieved_precision", meta["threshold_metrics"]["achieved_precision"])
        mlflow.log_metric("achieved_recall", meta["threshold_metrics"]["achieved_recall"])

        if "feature_importances" in meta:
            for feature, importance in meta["feature_importances"].items():
                mlflow.log_param(f"feature_importance_{feature}", importance)

        if model_type == "LogisticRegression":
            mlflow.sklearn.log_model(model, "model", registered_model_name=f"FraudDetection_{model_type}")
        elif model_type == "RandomForest":
            mlflow.sklearn.log_model(model, "model", registered_model_name=f"FraudDetection_{model_type}")
        elif model_type == "XGBoost":
            mlflow.xgboost.log_model(model, "model", registered_model_name=f"FraudDetection_{model_type}")

        mlflow.log_artifacts("artifacts/", artifact_path="artifacts")
        mlflow.log_text(json.dumps(meta, indent=2), "metadata.json")

        run_id = mlflow.active_run().info.run_id
        print(f"Model logged to MLflow with run_id: {run_id}")

        return run_id


def promote_model_to_stage(model_name: str, stage: str = "Production"):
    """Promote a model to a specific stage in MLflow.

    Args:
        model_name: Name of the registered model
        stage: Stage to promote to (Staging, Production, Archived)
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["None"])

    if versions:
        latest_version = versions[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage=stage
        )
        print(f"Model {model_name} version {latest_version} promoted to {stage}")
    else:
        print(f"No versions found for model {model_name}")


def load_model_from_mlflow(model_name: str, stage: str = "Production"):
    """Load a model from MLflow registry.

    Args:
        model_name: Name of the registered model
        stage: Model stage to load from

    Returns:
        Loaded model
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    print(f"Loaded model {model_name} from stage {stage}")

    return model


def get_best_model():
    """Get the best performing model based on ROC-AUC."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    models = ["FraudDetection_LogisticRegression", "FraudDetection_RandomForest", "FraudDetection_XGBoost"]

    best_model = None
    best_score = 0

    for model_name in models:
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                run_id = versions[0].run_id
                run = client.get_run(run_id)
                roc_auc = run.data.metrics.get("roc_auc", 0)

                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = model_name
        except Exception as e:
            print(f"Error checking {model_name}: {e}")

    return best_model, best_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow model management")
    parser.add_argument("--log", action="store_true", help="Log current model to MLflow")
    parser.add_argument("--promote", type=str, help="Promote model to stage")
    parser.add_argument("--load", type=str, help="Load model from MLflow")
    parser.add_argument("--best", action="store_true", help="Get best model")

    args = parser.parse_args()

    if args.log:
        model = joblib.load("artifacts/model.joblib")
        with open("artifacts/model_meta.json", 'r') as f:
            meta = json.load(f)
        model_type = meta.get("model_type", "LogisticRegression")
        log_model_to_mlflow(model, model_type, None, None, meta)

    elif args.promote:
        promote_model_to_stage(args.promote)

    elif args.load:
        load_model_from_mlflow(args.load)

    elif args.best:
        best_model, score = get_best_model()
        print(f"Best model: {best_model} with ROC-AUC: {score:.4f}")
