# Release Checklist (v1.0.0)

Follow this checklist to verify the system or prepare a new release.

## 1. Clean Environment
- [ ] Create a fresh virtual environment: `python3 -m venv .venv_test`
- [ ] Activate: `source .venv_test/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`

## 2. Pipeline Execution
Run the following commands in order and ensure no errors:
- [ ] `python src/fraudml/data/download.py` (Should create `data/raw.csv`)
- [ ] `python src/fraudml/data/preprocess.py` (Should create splits in `data/`)
- [ ] `python src/fraudml/models/train.py` (Should save `artifacts/model.joblib`)
- [ ] `python src/fraudml/models/evaluate.py` (Should save `reports/metrics.json`)

## 3. Automated Testing
- [ ] Run `pytest`
- [ ] Verify all tests pass (smoke + API tests).

## 4. Manual Verification
### Backend
- [ ] Start server: `uvicorn src.fraudml.api.app:app --reload`
- [ ] Visit `http://localhost:8000/health` -> {"status": "ok"}
- [ ] Visit `http://localhost:8000/docs` -> Swagger UI loads.

### Frontend
- [ ] Start dashboard: `streamlit run src/fraudml/ui/dashboard.py`
- [ ] Dashboard loads at `http://localhost:8501`.
- [ ] Adjust sliders -> "Analyzing..." spinner appears -> Result updates.

## 5. Deployment Info
- [ ] Ensure `DEPLOYMENT.md` is up to date.
- [ ] If deploying to Render/Streamlit Cloud, ensure `requirements.txt` is committed.

## Troubleshooting
- **Result**: `Address already in use`?
  - **Fix**: `lsof -ti:8000 | xargs kill -9`
- **Result**: `Model not loaded`?
  - **Fix**: Run `src/fraudml/models/train.py` first.
