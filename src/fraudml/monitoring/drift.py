import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift between reference and current distributions."""

    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_reference_stats()

    def _compute_reference_stats(self) -> Dict[str, dict]:
        """Compute statistics for reference distribution."""
        stats_dict = {}
        for col in self.reference_data.columns:
            if self.reference_data[col].dtype in ['float64', 'int64']:
                stats_dict[col] = {
                    'mean': self.reference_data[col].mean(),
                    'std': self.reference_data[col].std(),
                    'quantiles': self.reference_data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                stats_dict[col] = {
                    'value_counts': self.reference_data[col].value_counts(normalize=True).to_dict()
                }
        return stats_dict

    def _compute_ks_statistic(self, reference: pd.Series, current: pd.Series) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        return stats.ks_2samp(reference, current).statistic

    def _compute_population_stability_index(self, reference: pd.Series,
                                            current: pd.Series,
                                            buckets: int = 10) -> float:
        """Compute Population Stability Index (PSI)."""
        ref_quantiles = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        ref_quantiles = np.unique(ref_quantiles)

        ref_counts = []
        current_counts = []

        for i in range(len(ref_quantiles) - 1):
            if i == len(ref_quantiles) - 2:
                ref_mask = (ref_quantiles[i] <= reference) & (reference <= ref_quantiles[i + 1])
                current_mask = (ref_quantiles[i] <= current) & (current <= ref_quantiles[i + 1])
            else:
                ref_mask = (ref_quantiles[i] <= reference) & (reference < ref_quantiles[i + 1])
                current_mask = (ref_quantiles[i] <= current) & (current < ref_quantiles[i + 1])

            ref_counts.append(ref_mask.sum() / len(reference))
            current_counts.append(current_mask.sum() / len(current))

        ref_counts = np.array(ref_counts)
        current_counts = np.array(current_counts)

        ref_counts = np.where(ref_counts == 0, 0.0001, ref_counts)
        current_counts = np.where(current_counts == 0, 0.0001, current_counts)

        psi = np.sum((current_counts - ref_counts) * np.log(current_counts / ref_counts))
        return psi

    def check_drift(self, current_data: pd.DataFrame) -> Dict[str, dict]:
        """Check for drift in current data compared to reference.

        Args:
            current_data: Current data distribution

        Returns:
            Dictionary with drift results for each feature
        """
        results = {}

        for col in self.reference_data.columns:
            if col not in current_data.columns:
                results[col] = {
                    'drift_detected': True,
                    'message': 'Column missing in current data',
                    'drift_score': 1.0
                }
                continue

            if self.reference_data[col].dtype in ['float64', 'int64']:
                ref_series = self.reference_data[col].dropna()
                curr_series = current_data[col].dropna()

                if len(ref_series) == 0 or len(curr_series) == 0:
                    results[col] = {
                        'drift_detected': False,
                        'message': 'Insufficient data',
                        'drift_score': 0.0
                    }
                    continue

                ks_stat = self._compute_ks_statistic(ref_series, curr_series)
                psi = self._compute_population_stability_index(ref_series, curr_series)

                drift_score = max(ks_stat, psi / 10)
                drift_detected = drift_score > self.threshold

                results[col] = {
                    'drift_detected': drift_detected,
                    'ks_statistic': float(ks_stat),
                    'psi': float(psi),
                    'drift_score': float(drift_score),
                    'ref_mean': float(ref_series.mean()),
                    'curr_mean': float(curr_series.mean()),
                    'ref_std': float(ref_series.std()),
                    'curr_std': float(curr_series.std()),
                    'message': f"Drift detected: {drift_detected}"
                }

            else:
                ref_counts = self.reference_data[col].value_counts(normalize=True)
                curr_counts = current_data[col].value_counts(normalize=True)

                all_values = set(ref_counts.keys()) | set(curr_counts.keys())
                ref_array = np.array([ref_counts.get(v, 0) for v in all_values])
                curr_array = np.array([curr_counts.get(v, 0) for v in all_values])

                ref_array = np.where(ref_array == 0, 0.0001, ref_array)
                curr_array = np.where(curr_array == 0, 0.0001, curr_array)

                psi = np.sum((curr_array - ref_array) * np.log(curr_array / ref_array))
                drift_detected = psi > self.threshold

                results[col] = {
                    'drift_detected': drift_detected,
                    'psi': float(psi),
                    'drift_score': float(psi),
                    'message': f"Drift detected: {drift_detected}"
                }

        return results

    def get_overall_drift_status(self, drift_results: Dict[str, dict]) -> Tuple[bool, float]:
        """Get overall drift status.

        Args:
            drift_results: Results from check_drift

        Returns:
            Tuple of (drift_detected, max_drift_score)
        """
        drift_scores = []
        for col, result in drift_results.items():
            if 'drift_score' in result:
                drift_scores.append(result['drift_score'])

        if not drift_scores:
            return False, 0.0

        max_score = max(drift_scores)
        drift_detected = max_score > self.threshold

        return drift_detected, max_score


def detect_drift(reference_data: pd.DataFrame,
                 current_data: pd.DataFrame,
                 threshold: float = 0.1) -> Dict:
    """Convenience function for drift detection."""
    detector = DriftDetector(reference_data, threshold)
    drift_results = detector.check_drift(current_data)
    drift_detected, max_score = detector.get_overall_drift_status(drift_results)

    return {
        'drift_detected': drift_detected,
        'max_drift_score': max_score,
        'threshold': threshold,
        'feature_results': drift_results
    }


def log_drift_report(drift_results: Dict, output_path: str = None):
    """Log and optionally save drift report."""
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION REPORT")
    logger.info("=" * 60)
    logger.info(f"Overall Drift Detected: {drift_results['drift_detected']}")
    logger.info(f"Max Drift Score: {drift_results['max_drift_score']:.4f}")
    logger.info("-" * 60)

    for feature, result in drift_results['feature_results'].items():
        status = "DRIFT" if result['drift_detected'] else "OK"
        logger.info(f"  {feature}: {status} (score: {result.get('drift_score', 0):.4f})")

    logger.info("=" * 60)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(drift_results, f, indent=2)
        logger.info(f"Drift report saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect data drift")
    parser.add_argument("--reference", required=True, help="Path to reference data CSV")
    parser.add_argument("--current", required=True, help="Path to current data CSV")
    parser.add_argument("--output", default="reports/drift_report.json",
                        help="Path to save drift report")

    args = parser.parse_args()

    ref_data = pd.read_csv(args.reference)
    curr_data = pd.read_csv(args.current)

    results = detect_drift(ref_data, curr_data)
    log_drift_report(results, args.output)
