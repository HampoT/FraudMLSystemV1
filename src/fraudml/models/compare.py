import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def compare_models(data_dir, artifacts_dir, reports_dir):
    """Train and compare all model types."""
    from .train import train_model
    from .evaluate import evaluate_model

    models = ['LogisticRegression', 'RandomForest', 'XGBoost']
    results = {}

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print("=" * 60)

        model_artifacts = os.path.join(artifacts_dir, model_name.lower())
        os.makedirs(model_artifacts, exist_ok=True)

        model, meta = train_model(
            data_dir=data_dir,
            artifacts_dir=model_artifacts,
            model_type=model_name
        )

        results[model_name] = {
            'roc_auc': meta['metrics_val']['roc_auc'],
            'pr_auc': meta['metrics_val']['pr_auc'],
            'cv_roc_auc': meta['metrics_val']['cv_roc_auc_mean'],
            'cv_std': meta['metrics_val']['cv_roc_auc_std'],
            'threshold': meta['threshold'],
            'precision': meta['threshold_metrics']['achieved_precision'],
            'recall': meta['threshold_metrics']['achieved_recall']
        }

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'PR-AUC':<12} {'CV-ROC':<12} {'Threshold':<10}")
    print("-" * 60)

    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['roc_auc']:<12.4f} {metrics['pr_auc']:<12.4f} "
              f"{metrics['cv_roc_auc']:<12.4f} {metrics['threshold']:<10.4f}")

    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    print(f"\nBest model by ROC-AUC: {best_model}")

    os.makedirs(reports_dir, exist_ok=True)

    comparison_df = pd.DataFrame(results).T
    comparison_df.to_csv(os.path.join(reports_dir, 'model_comparison.csv'))
    print(f"Comparison saved to {os.path.join(reports_dir, 'model_comparison.csv')}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    model_names = list(results.keys())
    roc_aucs = [results[m]['roc_auc'] for m in model_names]
    ax1.bar(model_names, roc_aucs, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_title('ROC-AUC Comparison', fontsize=14)
    ax1.set_ylabel('ROC-AUC')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(roc_aucs):
        ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

    ax2 = axes[0, 1]
    pr_aucs = [results[m]['pr_auc'] for m in model_names]
    ax2.bar(model_names, pr_aucs, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax2.set_title('PR-AUC Comparison', fontsize=14)
    ax2.set_ylabel('PR-AUC')
    ax2.set_ylim(0, max(pr_aucs) * 1.2)
    for i, v in enumerate(pr_aucs):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)

    ax3 = axes[1, 0]
    x = np.arange(len(model_names))
    width = 0.35
    recalls = [results[m]['recall'] for m in model_names]
    precisions = [results[m]['precision'] for m in model_names]
    ax3.bar(x - width/2, recalls, width, label='Recall', color='#3498db')
    ax3.bar(x + width/2, precisions, width, label='Precision', color='#2ecc71')
    ax3.set_title('Precision vs Recall (at target)', fontsize=14)
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.legend()
    ax3.set_ylim(0, 1.1)

    ax4 = axes[1, 1]
    cv_means = [results[m]['cv_roc_auc'] for m in model_names]
    cv_stds = [results[m]['cv_std'] for m in model_names]
    ax4.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
            color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax4.set_title('Cross-Validation ROC-AUC', fontsize=14)
    ax4.set_ylabel('Mean ROC-AUC')
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    comparison_plot_path = os.path.join(reports_dir, 'model_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {comparison_plot_path}")

    return results


if __name__ == "__main__":
    compare_models(
        data_dir="data",
        artifacts_dir="artifacts",
        reports_dir="reports"
    )
