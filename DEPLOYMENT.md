# Deployment Guide

This guide explains how to deploy the **Fraud Detection ML System** so it's accessible publicly, not just on localhost.

We will use a standard "decoupled" architecture:
1.  **Backend (FastAPI)**: Deployed on a cloud service (e.g., Render or Railway).
2.  **Frontend (Streamlit)**: Deployed on Streamlit Cloud.

---

## 🚀 Part 1: Deploy Backend (The API)

We will use **Render** (free tier available) to host the FastAPI backend.

### 1. Prepare for Cloud
Ensure dependencies are in `requirements.txt`:
```bash
# Must include: fastapi, uvicorn, scikit-learn, pandas, numpy, joblib
cat requirements.txt
```

### 2. Push to GitHub
Make sure your code is committed and pushed to a public GitHub repository.

### 3. Deploy to Render
1.  Go to [dashboard.render.com](https://dashboard.render.com/).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  **Configuration**:
    *   **Name**: `fraud-api` (example)
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn src.fraudml.api.app:app --host 0.0.0.0 --port 10000`
5.  Click **Create Web Service**.

Wait for the build to finish. You will get a URL like `https://fraud-api.onrender.com`.

**Important**:
*   Note this URL. You need it for the frontend.
*   Once live, visit `https://YOUR-URL.onrender.com/docs` to verify it works.

---

## 🎨 Part 2: Deploy Frontend (The Dashboard)

We will use **Streamlit Cloud** (free, easiest for Streamlit apps).

### 1. Update API URL
In `src/fraudml/ui/dashboard.py`, you need to point the dashboard to your *live* backend instead of localhost.

**Option A (Hardcode)**:
Change line 6 in `dashboard.py`:
```python
API_URL = "https://fraud-api.onrender.com/predict" # Use your actual Render URL
```

**Option B (Environment Variable - Recommended)**:
Use `os.getenv` in python and set the variable in Streamlit Cloud secrets.

### 2. Deploy to Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **New App**.
3.  Select your GitHub repo, branch, and file path: `src/fraudml/ui/dashboard.py`.
4.  Click **Deploy**.

Your professional dashboard is now live on the internet! 🌍

---

## ⚡ Option 3: Quick Local Sharing (ngrok)

If you just want to show a friend *right now* without deploying:

1.  Start your backend locally: `uvicorn src.fraudml.api.app:app --reload`
2.  Start your frontend locally: `streamlit run src/fraudml/ui/dashboard.py`
3.  Use **ngrok** to tunnel your Streamlit port (8501):
    ```bash
    ngrok http 8501
    ```
    (Requires installing ngrok). This gives you a temporary public URL to send to anyone.
