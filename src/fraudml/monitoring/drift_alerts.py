import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    feature: str
    drift_detected: bool
    drift_score: float
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    timestamp: str


class DriftDetector:
    """Statistical drift detection for model monitoring."""

    def __init__(self, threshold: float = 0.1, window_size: int = 1000):
        self.threshold = threshold
        self.window_size = window_size
        self.reference_stats: Dict[str, Dict] = {}
        self.drift_history: List[DriftResult] = []

    def compute_reference_stats(self, reference_data: pd.DataFrame):
        """Compute statistics from reference dataset."""
        logger.info(f"Computing reference statistics from {len(reference_data)} samples")
        
        for col in reference_data.columns:
            if reference_data[col].dtype in ['float64', 'int64']:
                self.reference_stats[col] = {
                    'mean': float(reference_data[col].mean()),
                    'std': float(reference_data[col].std()),
                    'quantiles': {
                        'q25': float(reference_data[col].quantile(0.25)),
                        'q50': float(reference_data[col].quantile(0.5)),
                        'q75': float(reference_data[col].quantile(0.75))
                    },
                    'min': float(reference_data[col].min()),
                    'max': float(reference_data[col].max()),
                    'n': len(reference_data)
                }
            else:
                value_counts = reference_data[col].value_counts(normalize=True).to_dict()
                self.reference_stats[col] = {
                    'value_counts': value_counts,
                    'n': len(reference_data)
                }
        
        logger.info(f"Computed reference statistics for {len(self.reference_stats)} features")

    def _ks_test(self, reference: pd.Series, current: pd.Series) -> float:
        """Kolmogorov-Smirnov test statistic."""
        from scipy import stats
        return stats.ks_2samp(reference, current).statistic

    def _psi(self, reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
        """Population Stability Index."""
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
        return float(psi)

    def check_drift(self, current_data: pd.DataFrame) -> Tuple[bool, List[DriftResult]]:
        """Check for drift between reference and current data."""
        results = []
        any_drift = False
        
        for col in self.reference_stats.keys():
            if col not in current_data.columns:
                results.append(DriftResult(
                    feature=col,
                    drift_detected=True,
                    drift_score=1.0,
                    reference_mean=0,
                    current_mean=0,
                    reference_std=0,
                    current_std=0,
                    timestamp=datetime.utcnow().isoformat()
                ))
                any_drift = True
                continue
            
            ref_stats = self.reference_stats[col]
            
            if 'mean' in ref_stats:
                ref_series = ref_stats.get('mean', 0)
                ref_std = ref_stats.get('std', 0)
                current_series = current_data[col].dropna()
                current_mean = float(current_series.mean()) if len(current_series) > 0 else 0
                current_std = float(current_series.std()) if len(current_series) > 0 else 0
                
                ks_stat = self._ks_test(
                    pd.Series([ref_series] * 100), 
                    current_series
                ) if len(current_series) > 0 else 0
                
                psi = self._psi(
                    pd.Series([ref_series] * min(100, self.window_size)),
                    current_series.head(self.window_size)
                ) if len(current_series) > 0 else 0
                
                drift_score = max(ks_stat, psi / 10)
                drift_detected = drift_score > self.threshold
                
                if drift_detected:
                    any_drift = True
                
                results.append(DriftResult(
                    feature=col,
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                    reference_mean=ref_stats['mean'],
                    current_mean=current_mean,
                    reference_std=ref_std,
                    current_std=current_std,
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        self.drift_history.extend(results)
        
        return any_drift, results

    def get_drift_summary(self) -> Dict:
        """Get summary of drift status."""
        if not self.drift_history:
            return {"status": "no_data", "message": "No drift data collected"}
        
        drift_count = sum(1 for r in self.drift_history if r.drift_detected)
        total_count = len(self.drift_history)
        
        return {
            "status": "drift_detected" if drift_count > 0 else "stable",
            "features_with_drift": drift_count,
            "total_features_checked": total_count,
            "drift_rate": drift_count / max(1, total_count),
            "last_check": self.drift_history[-1].timestamp if self.drift_history else None
        }


class AlertManager:
    """Manage alerts for drift and performance issues."""

    def __init__(self):
        self.alert_channels: Dict[str, List[callable]] = {
            "slack": [],
            "email": [],
            "webhook": [],
            "log": []
        }
        self.alert_history: List[Dict] = []

    def add_slack_alert(self, webhook_url: str, channel: str = "#alerts"):
        """Add Slack webhook for alerts."""
        def slack_alert(message: Dict):
            import requests
            try:
                requests.post(webhook_url, json={
                    "channel": channel,
                    "text": f"Fraud Detection Alert: {message.get('title', 'Alert')}",
                    "attachments": [{
                        "color": message.get("severity", "warning"),
                        "fields": [
                            {"title": k, "value": str(v), "short": True}
                            for k, v in message.get("details", {}).items()
                        ]
                    }]
                })
            except Exception as e:
                logger.error(f"Slack alert failed: {e}")
        self.alert_channels["slack"].append(slack_alert)

    def add_email_alert(self, smtp_config: Dict, recipients: List[str]):
        """Add email alerts."""
        def email_alert(message: Dict):
            import smtplib
            from email.mime.text import MIMEText
            try:
                msg = MIMEText(json.dumps(message, indent=2))
                msg["Subject"] = f"Fraud Detection Alert: {message.get('title', 'Alert')}"
                msg["From"] = smtp_config["from"]
                msg["To"] = ", ".join(recipients)
                
                with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
                    server.starttls()
                    server.login(smtp_config["user"], smtp_config["password"])
                    server.sendmail(smtp_config["from"], recipients, msg.as_string())
            except Exception as e:
                logger.error(f"Email alert failed: {e}")
        self.alert_channels["email"].append(email_alert)

    def trigger_alert(self, title: str, severity: str, details: Dict):
        """Trigger alert on all channels."""
        alert = {
            "title": title,
            "severity": severity,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.alert_history.append(alert)
        logger.warning(f"Alert triggered: {title} ({severity})")
        
        for channel, handlers in self.alert_channels.items():
            for handler in handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed ({channel}): {e}")

    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get recent alerts."""
        return self.alert_history[-limit:]


def check_drift_and_alert(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame = None,
    detector: DriftDetector = None,
    alert_manager: AlertManager = None,
    auto_rollback_threshold: float = 0.2
) -> Dict:
    """Check for drift and trigger alerts if needed."""
    if detector is None:
        detector = DriftDetector(threshold=0.1)
    
    if not detector.reference_stats and reference_data is not None:
        detector.compute_reference_stats(reference_data)
    
    if not detector.reference_stats:
        return {"status": "no_reference", "message": "No reference statistics available"}
    
    drift_detected, results = detector.check_drift(current_data)
    
    drift_summary = detector.get_drift_summary()
    
    if drift_detected and alert_manager:
        drift_features = [r.feature for r in results if r.drift_detected]
        alert_manager.trigger_alert(
            title=f"Data drift detected in {len(drift_features)} features",
            severity="warning",
            details={
                "features": drift_features,
                "drift_summary": drift_summary,
                "scores": {r.feature: r.drift_score for r in results}
            }
        )
        
        high_drift_features = [r for r in results if r.drift_score > auto_rollback_threshold]
        if high_drift_features:
            alert_manager.trigger_alert(
                title="High drift detected - model rollback recommended",
                severity="critical",
                details={
                    "features": [r.feature for r in high_drift_features],
                    "threshold": auto_rollback_threshold
                }
            )
    
    return {
        "status": drift_summary["status"],
        "drift_detected": drift_detected,
        "features_with_drift": drift_summary["features_with_drift"],
        "results": [ {
            "feature": r.feature,
            "drift_detected": r.drift_detected,
            "drift_score": r.drift_score
        } for r in results]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drift detection")
    parser.add_argument("--reference", required=True, help="Reference data CSV")
    parser.add_argument("--current", required=True, help="Current data CSV")
    parser.add_argument("--output", default="reports/drift_report.json", help="Output path")
    args = parser.parse_args()
    
    ref_data = pd.read_csv(args.reference)
    curr_data = pd.read_csv(args.current)
    
    detector = DriftDetector()
    detector.compute_reference_stats(ref_data)
    drift_detected, results = detector.check_drift(curr_data)
    
    summary = detector.get_drift_summary()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "feature": r.feature,
                    "drift_detected": r.drift_detected,
                    "drift_score": r.drift_score,
                    "reference_mean": r.reference_mean,
                    "current_mean": r.current_mean
                } for r in results
            ],
            "timestamp": datetime.utcnow().isoformat()
        }, f, indent=2)
    
    print(f"Drift report saved to {args.output}")
    print(f"Status: {summary['status']}")
