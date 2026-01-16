# Production Deployment Guide

This guide covers deploying the Fraud Detection ML System to production.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Production Architecture                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Users     │───▶│   CDN/      │───▶│   K8s       │  │
│  │             │    │   WAF       │    │   Ingress   │  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘  │
│                                               │         │
│                    ┌──────────────────────────┘         │
│                    ▼                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │              K8s Cluster (EKS/GKE/AKS)          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │   │
│  │  │Fraud API  │  │Dashboard  │  │ MLflow    │   │   │
│  │  │(3+ pods)  │  │(2+ pods)  │  │(1 pod)    │   │   │
│  │  └───────────┘  └───────────┘  └───────────┘   │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │   │
│  │  │  Redis    │  │PostgreSQL │  │Prometheus │   │   │
│  │  │(Cache)    │  │(Data)     │  │(Metrics)  │   │   │
│  │  └───────────┘  └───────────┘  └───────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
│                    │                                    │
│                    ▼                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Cloud Services                      │   │
│  │  • S3 (model artifacts)  • CloudWatch (logs)    │   │
│  │  • Secrets Manager (API keys)  • Route53 (DNS)  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- Kubernetes cluster (EKS, GKE, or AKS)
- Helm 3.x
- kubectl configured
- Docker
- SSL certificate (Let's Encrypt or purchased)

## Quick Start (Docker Compose)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace fraud-detection
kubectl config set-context --current --namespace=fraud-detection
```

### 2. Deploy Infrastructure
```bash
# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=fraud-redis --timeout=120s
kubectl wait --for=condition=ready pod -l app=fraud-db --timeout=120s
```

### 3. Deploy API
```bash
# Build and push images
docker build -t fraud-ml-system:latest .
docker tag fraud-ml-system:latest ghcr.io/yourusername/fraud-ml-system:latest
docker push ghcr.io/yourusername/fraud-ml-system:latest

# Deploy API
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get pods -l app=fraud-api
kubectl logs -l app=fraud-api --tail=100
```

### 4. Deploy Dashboard
```bash
docker build -f Dockerfile.dashboard -t fraud-ml-dashboard:latest .
kubectl apply -f k8s/dashboard.yaml
```

### 5. Configure Ingress
```bash
# Edit dashboard.yaml to add your domain
# Then apply
kubectl apply -f k8s/dashboard.yaml
```

## Environment Configuration

### Production Secrets
```bash
# Create secrets
kubectl create secret generic fraud-secrets \
  --from-literal=API_KEY=your-production-api-key \
  --from-literal=DATABASE_URL=postgresql://user:pass@host:5432/db \
  --from-literal=REDIS_URL=redis://redis-host:6379 \
  --namespace=fraud-detection
```

### ConfigMap
```bash
kubectl create configmap fraud-config \
  --from-literal=LOG_LEVEL=INFO \
  --from-literal=PROMETHEUS_ENABLED=true \
  --from-literal=MLFLOW_TRACKING_URI=http://mlflow:5000 \
  --namespace=fraud-detection
```

## Scaling

### Horizontal Pod Autoscaler
```bash
# HPA is already configured in deployment.yaml
# Verify HPA
kubectl get hpa fraud-api

# Manual scaling
kubectl scale deployment fraud-api --replicas=10
```

### Resource Limits
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## Monitoring

### Access Grafana
```bash
kubectl port-forward svc/grafana 3000:3000
# Open http://localhost:3000
# Default credentials: admin/admin
```

### Access Prometheus
```bash
kubectl port-forward svc/prometheus 9090:9090
# Open http://localhost:9090
```

### Key Metrics
- `fraud_predictions_total`: Total predictions made
- `fraud_prediction_latency_seconds`: API latency
- `fraud_api_requests_in_progress`: In-flight requests

## MLflow Setup

### Access MLflow
```bash
kubectl port-forward svc/mlflow 5000:5000
# Open http://localhost:5000
```

### Register a Model
```python
from src.fraudml.models.mlflow_registry import log_model_to_mlflow

log_model_to_mlflow(model, "XGBoost", X_train, y_train, meta)
```

### Promote to Production
```python
from src.fraudml.models.mlflow_registry import promote_model_to_stage

promote_model_to_stage("FraudDetection_XGBoost", "Production")
```

## A/B Testing

### Configure Traffic Split
```python
from src.fraudml.models.ab_test import create_default_ab_config

config = create_default_ab_config()
router = ABTestRouter(config)
```

### Monitor Results
- Check MLflow for per-variant metrics
- Compare ROC-AUC between variants
- Monitor conversion rates

## Load Testing

### Run Locust
```bash
locust -f scripts/load_test.py \
  --host=http://fraud-api \
  --users=100 \
  --spawn-rate=10 \
  --run-time=5m
```

### View Results
- Open http://localhost:8089
- Monitor RPS, response times, failures

## Troubleshooting

### API Pods Not Starting
```bash
kubectl describe pod -l app=fraud-api
kubectl logs -l app=fraud-api --previous
```

### High Latency
```bash
# Check resource usage
kubectl top pods -n fraud-detection

# Check metrics
kubectl get --raw=/metrics | grep fraud_prediction_latency
```

### Database Connection Issues
```bash
kubectl exec -it svc/fraud-db -- psql -U postgres -d fraud_db
```

## Backup and Recovery

### PostgreSQL Backup
```bash
kubectl exec fraud-db -- pg_dump -U postgres fraud_db > backup.sql
```

### Restore
```bash
kubectl exec -i fraud-db -- psql -U postgres fraud_db < backup.sql
```

### Model Artifacts
```bash
# Backup to S3
aws s3 sync artifacts/ s3://your-backup-bucket/artifacts/
```

## Security

### API Authentication
- All `/v1/*` endpoints require `X-API-Key` header
- Rate limiting: 100 req/min per key
- Use HTTPS only in production

### Secrets Management
- Use Kubernetes Secrets or external secrets manager
- Rotate API keys regularly
- Enable audit logging

### Network Security
- Restrict ingress sources
- Use VPC peering for databases
- Enable encryption at rest
