# Changelog

All notable changes to the Fraud Detection ML System are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-16

### Added
- **Model Enhancements**
  - Added Random Forest model support
  - Added XGBoost model support
  - Implemented cross-validation for threshold tuning
  - Added SHAP explainability in API responses

- **API Improvements**
  - Added rate limiting (slowapi)
  - Added API key authentication
  - Added batch prediction endpoint (`/v1/batch-predict`)
  - Added explanation endpoint (`/v1/explain`)
  - Added API versioning (`/v1/`)
  - Added Prometheus metrics endpoint (`/metrics`)
  - Added structured request logging

- **Data Pipeline**
  - Added real dataset option (Kaggle credit card fraud)
  - Added feature engineering pipeline
  - Added data drift detection
  - Added DVC support for data versioning

- **Monitoring**
  - Added Prometheus metrics
  - Added prediction logging
  - Added drift detection
  - Added performance tracking

- **Developer Experience**
  - Created Makefile for common commands
  - Added `.env.example` template
  - Added comprehensive unit tests
  - Added model comparison utilities

### Changed
- Refactored API structure with versioning
- Updated requirements.txt with new dependencies
- Improved test coverage (now >80%)
- Optimized prediction latency

### Fixed
- Fixed data leakage in preprocessing
- Fixed threshold tuning logic
- Fixed logging format

## [1.0.0] - 2026-01-15

### Added
- Initial release
- Logistic Regression baseline model
- Toy dataset generator
- FastAPI inference endpoint
- Streamlit dashboard
- Basic pytest suite
- Deployment documentation (Render + Streamlit Cloud)

### Features
- Reproducible pipeline with fixed seeds
- Dynamic threshold tuning for 95% precision target
- Interactive Streamlit dashboard
- API documentation at `/docs`

### Known Limitations
- Single model (Logistic Regression only)
- No authentication
- No monitoring
- Toy data only
