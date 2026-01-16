# Architecture Documentation

## Overview

The Fraud Detection ML System has evolved from a simple logistic regression model to a production-ready, scalable architecture supporting multiple models, monitoring, and enterprise features.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRAUD DETECTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   Client    │───▶│  Streamlit  │───▶│    FastAPI Backend     │  │
│  │  (Browser)  │    │  Dashboard  │    │    /v1/predict          │  │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘  │
│                                                     │                │
│                    ┌───────────────────────────────┘                │
│                    ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    API GATEWAY LAYER                        │    │
│  │  • Rate Limiting (slowapi)  • Authentication (API Key)      │    │
│  │  • Request Logging  • Versioning (/v1, /v2)                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  PREDICTION SERVICE                         │    │
│  │  • Model Registry (MLflow-compatible)  • Caching (Redis)    │    │
│  │  • SHAP Explanations  • Batch Predictions                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                             │                                        │
│         ┌───────────────────┼───────────────────┐                   │
│         ▼                   ▼                   ▼                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │
│  │Logistic     │    │Random       │    │XGBoost              │    │
│  │Regression   │    │Forest       │    │Gradient Boosting    │    │
│  └─────────────┘    └─────────────┘    └─────────────────────┘    │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  DATA PIPELINE LAYER                        │    │
│  │  • Raw Data → Preprocessing → Feature Engineering           │    │
│  │  • Train/Val/Test Split  • Data Drift Detection             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  DATA STORAGE                               │    │
│  │  • PostgreSQL (production)  • CSV Files (local dev)         │    │
│  │  • Model Artifacts (artifacts/)  • Reports (reports/)        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  MONITORING & OBSERVABILITY                  │    │
│  │  • Prometheus Metrics (/metrics)  • Structured Logging       │    │
│  │  • Performance Tracking  • Drift Detection                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

#### Raw Data Sources
- **Toy Dataset**: Generated locally via `src/fraudml/data/download.py`
- **Real Dataset**: Kaggle Credit Card Fraud Detection dataset
- **Production**: PostgreSQL with TimescaleDB extension for time-series

#### Feature Engineering
The system now includes engineered features:
- **Transaction Velocity**: Count and amount of recent transactions
- **Amount Z-Score**: How unusual is this transaction amount
- **Time Features**: Hour of day, day of week, is weekend
- **Historical Patterns**: Merchant risk scores, user spending patterns

### 2. Model Layer

#### Supported Models
1. **Logistic Regression** (baseline)
   - Fast, interpretable
   - Best for: Initial deployment, debugging

2. **Random Forest**
   - Handles non-linear relationships
   - Best for: Complex fraud patterns

3. **XGBoost**
   - Highest performance on tabular data
   - Best for: Production deployment

#### Model Registry
- Models are stored in `artifacts/` with versioned metadata
- Each model includes: training date, metrics, feature list, threshold
- Supports model comparison and rollback

### 3. API Layer

#### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/predict` | POST | Single prediction |
| `/v1/batch-predict` | POST | Batch predictions |
| `/v1/explain` | POST | Prediction with SHAP values |
| `/metrics` | GET | Prometheus metrics |
| `/logs` | GET | Query prediction logs |

#### Authentication
- API Key-based authentication
- Rate limiting: 100 requests/minute per key
- Keys configured via environment variables

### 4. Monitoring Layer

#### Metrics Exported
- `fraud_predictions_total`: Total predictions
- `fraud_positive_predictions`: Fraud flags raised
- `fraud_prediction_latency_seconds`: API latency
- `model_version`: Current model version

#### Drift Detection
- Monitors feature distribution shifts
- Alerts when drift exceeds threshold
- Triggers retraining pipeline

## Deployment Architecture

### Development
```
Local: uvicorn + streamlit
Data: CSV files
Cache: None
```

### Production
```
Backend: Kubernetes with HPA (3+ replicas)
Database: PostgreSQL + Redis cache
Deployment: Blue/Green with canary analysis
CI/CD: GitHub Actions → Docker Build → K8s Deploy
```

## Security

- All API endpoints secured with API key
- PII encryption at rest and in transit
- Audit logging for all predictions
- Rate limiting to prevent abuse
