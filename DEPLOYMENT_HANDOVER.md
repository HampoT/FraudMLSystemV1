# AI Assistant Deployment Hand-Off Prompt

## Copy and paste the following prompt to your AI assistant (Claude Code, ChatGPT, etc.):

---

**System Prompt:**

You are an expert DevOps engineer and MLOps specialist. Your task is to deploy the Fraud Detection ML System to Render.com (a cloud platform) based on the repository at `https://github.com/HampoT/FraudMLSystemV1`.

## Project Overview

A production-ready fraud detection ML system with:
- FastAPI backend for predictions
- Streamlit dashboard for visualization
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Feature engineering with 18+ engineered features
- SHAP explainability
- Prometheus metrics
- Docker containerization

## Current Project Structure

```
fraud-ml-system/
├── src/fraudml/
│   ├── api/app.py          # FastAPI application (port 8000)
│   ├── ui/dashboard.py     # Streamlit dashboard (port 8501)
│   ├── data/download.py    # Data generation
│   ├── data/preprocess.py  # Data preprocessing
│   ├── models/train.py     # Model training
│   ├── models/evaluate.py  # Model evaluation
│   └── monitoring/         # Metrics and drift detection
├── Dockerfile              # API container
├── Dockerfile.dashboard    # Dashboard container
├── docker-compose.yml      # Local full-stack deployment
├── k8s/                    # Kubernetes manifests
├── render.yaml             # Render configuration
├── requirements.txt        # Python dependencies
└── .env.example            # Environment template
```

## Deployment Requirements

### 1. Deploy Backend Service (FastAPI) to Render

**Settings:**
- **Service Name**: `fraud-ml-backend`
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn src.fraudml.api.app:app --host 0.0.0.0 --port $PORT`
- **Root Directory**: (leave empty or set to repo root)

**Environment Variables to Configure:**
```
MODEL_PATH=artifacts/model.joblib
META_PATH=artifacts/model_meta.json
API_KEY=secure-random-api-key-here
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

**Health Check:** Render's automatic health check on `/health` endpoint

### 2. Deploy Frontend Service (Streamlit Dashboard) to Render

**Settings:**
- **Service Name**: `fraud-ml-dashboard`
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run src/fraudml/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0`
- **Root Directory**: (leave empty or set to repo root)

**Environment Variables to Configure:**
```
BACKEND_URL=https://fraud-ml-backend.onrender.com
```

### 3. Automatic Model Training (Bootstrap)

The API is configured to automatically train the model on startup if artifacts are missing. Ensure the build process runs:
```
python src/fraudml/data/download.py
python src/fraudml/data/preprocess.py
python src/fraudml/models/train.py --model XGBoost
```

### 4. Build Timeout

Set **Build Timeout** to 20 minutes (model training takes time).

### 5. Health Check Endpoints

Verify these endpoints work:
- Backend: `https://fraud-ml-backend.onrender.com/health`
- API Docs: `https://fraud-ml-backend.onrender.com/docs`
- Dashboard: `https://fraud-ml-dashboard.onrender.com`

## Steps to Execute

### Step 1: Connect Repository
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository `HampoT/FraudMLSystemV1`
4. Select the branch (main)

### Step 2: Configure Backend Service
```
Name: fraud-ml-backend
Environment: Python 3
Region: Oregon (or closest to your users)
Branch: main
Build Command: pip install -r requirements.txt
Start Command: uvicorn src.fraudml.api.app:app --host 0.0.0.0 --port $PORT
```

### Step 3: Configure Environment Variables
Add the following in the "Environment Variables" section:
```
API_KEY=<generate-a-secure-random-string>
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

### Step 4: Deploy
1. Click "Create Web Service"
2. Wait for build and deployment (10-20 minutes)
3. Check logs for any errors

### Step 5: Deploy Dashboard
```
Name: fraud-ml-dashboard
Environment: Python 3
Region: Oregon
Branch: main
Build Command: pip install -r requirements.txt
Start Command: streamlit run src/fraudml/ui/dashboard.py --server.port $PORT --server.address 0.0.0.0
Environment Variables:
  BACKEND_URL=https://fraud-ml-backend.onrender.com
```

### Step 6: Update Dashboard Environment
After backend is deployed, update dashboard's `BACKEND_URL` to the actual backend URL.

### Step 7: Verify Deployment

Test the endpoints:

```bash
# Health check
curl https://fraud-ml-backend.onrender.com/health

# Single prediction
curl -X POST https://fraud-ml-backend.onrender.com/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "amount": 1500.00,
    "hour": 14,
    "device_score": 0.85,
    "country_risk": 2
  }'

# Batch prediction
curl -X POST https://fraud-ml-backend.onrender.com/v1/batch-predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "transactions": [
      {"amount": 100, "hour": 10, "device_score": 0.9, "country_risk": 1},
      {"amount": 5000, "hour": 3, "device_score": 0.1, "country_risk": 5}
    ]
  }'
```

## Troubleshooting

### Build Fails
- Check that Python 3.12 is selected
- Increase build timeout to 20 minutes
- Ensure `requirements.txt` has all dependencies

### Model Not Training
- Check that the download/preprocess/train commands run in build
- Verify sufficient memory for XGBoost training
- Check logs for specific errors

### Dashboard Can't Connect to API
- Verify `BACKEND_URL` is correct (no trailing slash)
- Ensure API is deployed and healthy
- Check CORS settings in API

### Slow Response Times
- Consider upgrading Render plan
- Implement Redis caching (see `src/fraudml/api/cache.py`)
- Monitor memory usage

## Rollback Plan

If deployment fails:
1. Go to Render dashboard
2. Select the service
3. Click "Rollback" to previous version
4. Or deploy from a previous commit:
   ```bash
   git checkout <previous-commit-hash>
   git push origin main
   ```

## Post-Deployment Tasks

1. **Set up monitoring**: Check `/metrics` endpoint
2. **Configure alerts**: Set up alerts for failed health checks
3. **Custom domain**: (Optional) Add custom domain in Render settings
4. **SSL**: Render automatically provides SSL via Let's Encrypt
5. **Scale up**: Increase instance type if needed for production traffic

## Repository Reference

- **Repository URL**: `https://github.com/HampoT/FraudMLSystemV1`
- **Render.yaml**: Already configured in repo root
- **Documentation**: See `/docs/` directory for full documentation

---

**End of Hand-Off Prompt**

## Quick Action Items for AI Assistant

1. Read the repository structure
2. Check `render.yaml` configuration
3. Verify `requirements.txt` is complete
4. Deploy backend service to Render
5. Deploy dashboard service to Render
6. Verify all endpoints work
7. Report any errors or issues

## Expected Outcome

After successful deployment, you should have:
- **Backend URL**: `https://fraud-ml-backend.onrender.com`
- **Dashboard URL**: `https://fraud-ml-dashboard.onrender.com`
- **API Docs**: `https://fraud-ml-backend.onrender.com/docs`

All systems should be operational with:
- Health check passing
- Predictions working
- Dashboard connected to API
- No errors in logs
