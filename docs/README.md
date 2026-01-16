# Fraud Detection ML System - Complete Documentation

## Overview

This comprehensive documentation covers the Fraud Detection ML System, a production-ready machine learning solution for detecting fraudulent transactions.

## Documentation Structure

### [Architecture](architecture/README.md)
- System architecture overview
- Component details (Data, Model, API, Monitoring layers)
- Deployment architecture
- Security considerations

### [API Reference](api/README.md)
- Authentication methods
- Rate limiting
- Endpoint documentation
- Request/response formats
- SDK examples
- Error codes

### [Development Guide](development/CONTRIBUTING.md)
- Getting started
- Development workflow
- Code style guidelines
- Adding new models
- Testing guidelines
- Pull request process

### [Scaling Guide](scaling/README.md)
- Scaling tiers (Development to Enterprise)
- Data layer scaling
- Compute layer scaling
- Model scaling strategies
- Performance optimization
- Monitoring at scale
- Cost optimization
- Disaster recovery

## Quick Links

### Getting Started
```bash
# Clone and setup
git clone https://github.com/your-org/fraud-ml-system.git
cd fraud-ml-system
make install-dev

# Run the full pipeline
make pipeline

# Start services
make dev
```

### Training Models
```bash
# Train Logistic Regression (baseline)
make train-lr

# Train Random Forest
make train-rf

# Train XGBoost
make train-xgb

# Train all and compare
make train-all
```

### API Usage
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/v1/predict",
    headers={"X-API-Key": "your-api-key"},
    json={
        "amount": 1500.00,
        "hour": 14,
        "device_score": 0.85,
        "country_risk": 2
    }
)
```

## Changelog

See [CHANGELOG.md](development/CHANGELOG.md) for version history and updates.

## Contributing

See [CONTRIBUTING.md](development/CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License.

---

## System Capabilities

### Models Supported
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Handles non-linear relationships
- **XGBoost**: Highest performance on tabular data

### Features
- Feature engineering (18+ features)
- SHAP explainability
- Dynamic threshold tuning
- Cross-validation
- Data drift detection
- Prometheus metrics
- Rate limiting
- API authentication

### Deployment Options
- Local development
- Render/Railway (backend)
- Streamlit Cloud (frontend)
- Kubernetes (production)
