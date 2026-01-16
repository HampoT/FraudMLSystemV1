# API Documentation

## Base URL

```
Development: http://localhost:8000
Production: https://your-api-domain.com
```

## Authentication

All API requests require an API key in the header:

```
X-API-Key: your-api-key-here
```

## Rate Limits

| Plan | Requests/Minute |
|------|-----------------|
| Free | 100 |
| Pro | 1000 |
| Enterprise | Unlimited |

## Endpoints

### Health Check

**GET** `/health`

Returns the health status of the API and model.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "2026-01-16T00:16:33.068343",
  "model_type": "LogisticRegression",
  "timestamp": "2026-01-16T01:27:00Z"
}
```

---

### Single Prediction (v1)

**POST** `/v1/predict`

Predict whether a single transaction is fraudulent.

**Request:**
```json
{
  "amount": 1500.00,
  "hour": 14,
  "device_score": 0.85,
  "country_risk": 2,
  "merchant_id": "MERCHANT_123",
  "user_id": "USER_456",
  "transaction_count_last_hour": 3
}
```

**Required Fields:**
| Field | Type | Description |
|-------|------|-------------|
| amount | float | Transaction amount in USD |
| hour | int | Hour of transaction (0-23) |
| device_score | float | Device reliability score (0-1) |
| country_risk | int | Country risk level (1-5) |

**Optional Fields:**
| Field | Type | Description |
|-------|------|-------------|
| merchant_id | string | Merchant identifier |
| user_id | string | User identifier |
| transaction_count_last_hour | int | Recent transaction count |

**Response:**
```json
{
  "fraud_probability": 0.0234,
  "fraud_label": 0,
  "model_version": "2026-01-16T00:16:33.068343",
  "model_type": "XGBoost",
  "threshold": 0.95,
  "prediction_id": "pred_abc123xyz",
  "processing_time_ms": 45.2,
  "timestamp": "2026-01-16T01:27:00Z"
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| fraud_probability | float | Probability of fraud (0-1) |
| fraud_label | int | 1 if fraud, 0 if legitimate |
| model_version | string | Version of model used |
| model_type | string | Model algorithm type |
| threshold | float | Decision threshold used |
| prediction_id | string | Unique ID for this prediction |
| processing_time_ms | float | Processing time in milliseconds |
| timestamp | string | ISO timestamp of prediction |

---

### Batch Prediction (v1)

**POST** `/v1/batch-predict`

Predict fraud for multiple transactions.

**Request:**
```json
{
  "transactions": [
    {
      "amount": 100.00,
      "hour": 10,
      "device_score": 0.9,
      "country_risk": 1
    },
    {
      "amount": 5000.00,
      "hour": 3,
      "device_score": 0.1,
      "country_risk": 5
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "fraud_probability": 0.0012,
      "fraud_label": 0
    },
    {
      "fraud_probability": 0.8923,
      "fraud_label": 1
    }
  ],
  "batch_id": "batch_xyz789",
  "total_processing_time_ms": 125.5,
  "model_version": "2026-01-16T00:16:33.068343",
  "timestamp": "2026-01-16T01:27:00Z"
}
```

---

### Explain Prediction (v1)

**POST** `/v1/explain`

Get prediction with SHAP feature importance values.

**Request:**
```json
{
  "amount": 2500.00,
  "hour": 2,
  "device_score": 0.15,
  "country_risk": 4
}
```

**Response:**
```json
{
  "fraud_probability": 0.8472,
  "fraud_label": 1,
  "explanation": {
    "base_value": 0.05,
    "feature_contributions": [
      {
        "feature": "amount",
        "value": 2500.00,
        "contribution": 0.32,
        "description": "High transaction amount increases fraud risk"
      },
      {
        "feature": "hour",
        "value": 2,
        "contribution": 0.15,
        "description": "Late night transactions are higher risk"
      },
      {
        "feature": "device_score",
        "value": 0.15,
        "contribution": 0.25,
        "description": "Low device trust score increases risk"
      },
      {
        "feature": "country_risk",
        "value": 4,
        "contribution": 0.12,
        "description": "High-risk country contributes to fraud probability"
      }
    ]
  },
  "model_version": "2026-01-16T00:16:33.068343",
  "timestamp": "2026-01-16T01:27:00Z"
}
```

---

### Prometheus Metrics

**GET** `/metrics`

Returns Prometheus-formatted metrics.

**Example Output:**
```
# HELP fraud_predictions_total Total number of fraud predictions
# TYPE fraud_predictions_total counter
fraud_predictions_total{model="XGBoost"} 12345

# HELP fraud_positive_predictions Total fraud predictions (fraud_label=1)
# TYPE fraud_positive_predictions counter
fraud_positive_predictions{model="XGBoost"} 234

# HELP fraud_prediction_latency_seconds Prediction latency in seconds
# TYPE fraud_prediction_latency_seconds histogram
fraud_prediction_latency_seconds_bucket{le="0.01"} 5000
fraud_prediction_latency_seconds_bucket{le="0.05"} 12000
fraud_prediction_latency_seconds_bucket{le="0.1"} 12300
fraud_prediction_latency_seconds_bucket{le="+Inf"} 12345
```

---

### Query Logs (Admin)

**GET** `/v1/logs`

Query prediction logs (requires admin API key).

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| start_time | ISO8601 | Start of time range |
| end_time | ISO8601 | End of time range |
| fraud_label | int | Filter by label |
| limit | int | Max results (default 100) |

**Response:**
```json
{
  "logs": [
    {
      "prediction_id": "pred_abc123",
      "timestamp": "2026-01-16T01:27:00Z",
      "fraud_probability": 0.8472,
      "fraud_label": 1,
      "amount": 2500.00,
      "user_id": "USER_456"
    }
  ],
  "total": 1
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid API key |
| 429 | Too Many Requests - Rate limit exceeded |
| 503 | Service Unavailable - Model not loaded |
| 500 | Internal Server Error |

## SDKs

### Python

```python
from fraudml import FraudDetectionClient

client = FraudDetectionClient(api_key="your-key")

# Single prediction
result = client.predict(
    amount=1500.00,
    hour=14,
    device_score=0.85,
    country_risk=2
)

# Batch prediction
results = client.batch_predict([
    {"amount": 100, "hour": 10, "device_score": 0.9, "country_risk": 1},
    {"amount": 5000, "hour": 3, "device_score": 0.1, "country_risk": 5}
])

# With explanation
result = client.explain(
    amount=2500.00,
    hour=2,
    device_score=0.15,
    country_risk=4
)
```

### JavaScript/TypeScript

```typescript
import { FraudDetectionClient } from '@fraud-ml/sdk';

const client = new FraudDetectionClient({
  apiKey: process.env.FRAUD_API_KEY
});

const result = await client.predict({
  amount: 1500,
  hour: 14,
  deviceScore: 0.85,
  countryRisk: 2
});
```
