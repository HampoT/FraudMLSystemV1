# Scaling Guide

## Overview

This guide covers strategies for scaling the Fraud Detection ML System from prototype to production with millions of transactions.

## Scaling Tiers

### Tier 1: Development (Current)
- **Volume**: < 1K requests/day
- **Data**: CSV files
- **Compute**: Local machine
- **Latency**: < 100ms

### Tier 2: Early Production
- **Volume**: 1K - 100K requests/day
- **Data**: PostgreSQL
- **Compute**: Single server (Render/Railway)
- **Latency**: < 50ms

### Tier 3: Scale-Up
- **Volume**: 100K - 10M requests/day
- **Data**: PostgreSQL + TimescaleDB
- **Compute**: Kubernetes (3-10 nodes)
- **Latency**: < 20ms

### Tier 4: Enterprise
- **Volume**: 10M+ requests/day
- **Data**: Data warehouse (Snowflake/BigQuery)
- **Compute**: Kubernetes (50+ nodes) + Serverless
- **Latency**: < 10ms

---

## Data Layer Scaling

### Current State (CSV)
```
data/
├── raw.csv          # 400KB
├── X_train.csv      # 250KB
├── X_val.csv        # 83KB
└── X_test.csv       # 83KB
```

### Migration Path

**Step 1: PostgreSQL**
```python
# src/fraudml/data/postgres_loader.py
import pandas as pd
from sqlalchemy import create_engine

def load_from_postgres(query):
    engine = create_engine(os.getenv("DATABASE_URL"))
    return pd.read_sql(query, engine)
```

**Step 2: TimescaleDB for Time-Series**
```python
# Enable TimescaleDB extension
# CREATE EXTENSION timescaledb;

# Create hypertable for transactions
# SELECT create_hypertable('transactions', 'timestamp');
```

**Step 3: Data Warehouse**
- Migrate to Snowflake/BigQuery for analytics
- Keep PostgreSQL for real-time predictions

### Data Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────▶│  Ingestion  │────▶│  Storage    │
│             │     │  (Kafka)    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Batch     │◀───▶│  Processing │◀───▶│   Feature   │
│   Training  │     │  (Spark)    │     │   Store     │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Compute Layer Scaling

### Horizontal Scaling

**Kubernetes Deployment:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-api
  template:
    spec:
      containers:
      - name: api
        image: fraud-ml-system:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Caching Layer

```python
# src/fraudml/api/cache.py
import redis
from functools import lru_cache

r = redis.Redis.from_url(os.getenv("REDIS_URL"))

def cached_prediction(transaction_hash):
    """Cache predictions for similar transactions."""
    cached = r.get(f"pred:{transaction_hash}")
    if cached:
        return json.loads(cached)
    return None

def set_prediction(transaction_hash, prediction, ttl=3600):
    r.setex(f"pred:{transaction_hash}", ttl, json.dumps(prediction))
```

**Caching Strategy:**
- Cache predictions by transaction hash
- TTL: 1 hour for similar amounts
- Cache miss: < 10% of requests

### Model Serving

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL SERVING LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Request   │───▶│   Load      │───▶│   Model Server  │  │
│  │             │     │   Balancer  │     │   (TorchServe) │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                               │              │
│                    ┌──────────────────────────┘              │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              MODEL ARTIFACT REGISTRY                │    │
│  │  • Model versions  • A/B test assignments           │    │
│  │  • Rollback capability  • Canary releases           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Scaling

### Model Versioning

```python
# src/fraudml/models/registry.py
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, tracking_uri):
        self.client = MlflowClient(tracking_uri)
    
    def register_model(self, name, artifacts_path):
        result = self.client.create_model_version(
            name=name,
            source=artifacts_path,
            run_id=run_id
        )
        return result
    
    def promote_model(self, name, stage="Production"):
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
```

### A/B Testing

```python
# src/fraudml/models/ab_test.py
import random

class ABTestRouter:
    def __init__(self, models, traffic_split={"lr": 0.2, "rf": 0.3, "xgb": 0.5}):
        self.models = models
        self.traffic_split = traffic_split
    
    def get_model(self, user_id=None):
        if user_id:
            # Consistent routing for same user
            hash_val = hash(user_id)
            rand = hash_val % 100
        else:
            rand = random.randint(0, 99)
        
        cumulative = 0
        for model, split in self.traffic_split.items():
            cumulative += split * 100
            if rand < cumulative:
                return self.models[model]
        
        return self.models["xgb"]
```

### Model Compression

```python
# For edge deployment
import onnx
from sklearn.ensemble import forest_to_fast

# Convert to ONNX for faster inference
onnx_model = convert_sklearn(model, "decision_tree", X_sample)
```

---

## Performance Optimization

### Async Processing

```python
# src/fraudml/api/async_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

@app.post("/v1/batch-predict")
async def batch_predict_batch(batch: BatchRequest):
    tasks = [predict_one(t) for t in batch.transactions]
    results = await asyncio.gather(*tasks)
    return {"predictions": results}
```

### Connection Pooling

```python
# Database connection pool
from sqlalchemy.pool import QueuePool

engine = create_engine(
    os.getenv("DATABASE_URL"),
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600
)
```

### Profiling

```python
# Add to prediction endpoint
import cProfile
import pstats

def profile_prediction():
    profiler = cProfile.Profile()
    profiler.enable()
    # ... prediction logic ...
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)
```

---

## Monitoring at Scale

### Metrics Collection

```python
# src/fraudml/monitoring/prometheus.py
from prometheus_client import Counter, Histogram, Gauge

PREDICTION_COUNTER = Counter(
    'fraud_predictions_total',
    'Total fraud predictions',
    ['model', 'label']
)

PREDICTION_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction latency',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FRAUD_RATE = Gauge(
    'fraud_detection_rate',
    'Current fraud detection rate',
    ['hour']
)
```

### Alerting

```yaml
# alerts/fraud-detection-alerts.yaml
groups:
- name: fraud-detection
  rules:
  - alert: HighFraudRate
    expr: rate(fraud_positive_predictions[5m]) > 100
    for: 5m
    annotations:
      summary: "High fraud rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(fraud_prediction_latency_seconds_bucket[5m])) > 0.5
    for: 5m
    annotations:
      summary: "API latency exceeds 500ms"
```

### Dashboards

```
Grafana Dashboard: Fraud Detection Overview
├── Total Predictions (last 24h)
├── Fraud Rate Over Time
├── Prediction Latency (p50, p95, p99)
├── Model Performance (ROC-AUC, PR-AUC)
├── Error Rate
├── API Key Usage
└── Resource Utilization (CPU, Memory)
```

---

## Cost Optimization

### Spot Instances

```yaml
# k8s/nodepool.yaml
nodePools:
- name: spot
  config:
    minNodeCount: 3
    maxNodeCount: 50
    spot: true
```

### Model Selection

| Model | Latency | Accuracy | Cost/Query |
|-------|---------|----------|------------|
| Logistic Regression | 1ms | Baseline | $0.0001 |
| Random Forest | 5ms | +5% | $0.0005 |
| XGBoost | 3ms | +10% | $0.0003 |
| Deep Learning | 10ms | +15% | $0.001 |

**Strategy**: Route low-risk transactions to faster models, high-risk to accurate models.

### Resource Allocation

```
Cost Breakdown (10M requests/month):
├── Compute: $500-1000/month
├── Database: $200-500/month
├── Storage: $50-100/month
├── Monitoring: $100-200/month
└── Total: $850-1800/month
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily database backup
0 2 * * * pg_dump fraud_db | gzip > /backup/fraud_db_$(date +\%Y\%m\%d).sql.gz

# Model artifact backup
0 3 * * * rclone sync artifacts/ s3://backup-artifacts/
```

### Failover

```
Primary Region: us-east-1
└── API: fraud-api-primary.onrender.com
└── DB: Primary PostgreSQL

Secondary Region: eu-west-1
└── API: fraud-api-secondary.onrender.com
└── DB: Read replica PostgreSQL

DNS: fraud-api.company.com → Cloudflare Load Balancer
     Health check every 30s, automatic failover
```
