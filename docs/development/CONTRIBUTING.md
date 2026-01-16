# Contributing Guide

## Getting Started

### Prerequisites

- Python 3.12+
- Git
- pip or uv

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/fraud-ml-system.git
   cd fraud-ml-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   # or
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run tests**
   ```bash
   make test
   ```

## Development Workflow

### Code Style

- **Python**: Black formatter, Ruff linter
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all modules and functions

### Running Commands

```bash
# Install development dependencies
make install-dev

# Run linter and formatter
make lint

# Run tests
make test

# Run tests with coverage
make coverage

# Train model
make train

# Start API
make api

# Start dashboard
make dashboard

# Full pipeline
make pipeline
```

### Running Services

**API Server:**
```bash
uvicorn src.fraudml.api.app:app --reload
```

**Streamlit Dashboard:**
```bash
streamlit run src/fraudml/ui/dashboard.py
```

**All Services (dev):**
```bash
make dev
```

## Project Structure

```
fraud-ml-system/
├── src/fraudml/
│   ├── api/              # FastAPI application
│   ├── data/             # Data processing
│   │   ├── download.py   # Data download/generation
│   │   ├── preprocess.py # Data preprocessing
│   │   └── features.py   # Feature engineering
│   ├── models/           # ML models
│   │   ├── train.py      # Model training
│   │   ├── evaluate.py   # Model evaluation
│   │   └── compare.py    # Model comparison
│   ├── monitoring/       # Monitoring & metrics
│   │   ├── metrics.py    # Prometheus metrics
│   │   └── drift.py      # Drift detection
│   └── ui/               # Dashboard
├── tests/                # Pytest tests
├── docs/                 # Documentation
├── artifacts/            # Model artifacts
├── data/                 # Data files
└── reports/              # Generated reports
```

## Adding New Models

1. Create new model file in `src/fraudml/models/`
2. Implement `train()` and `predict()` functions
3. Add model to `ModelRegistry` in `src/fraudml/models/train.py`
4. Add tests in `tests/`
5. Update documentation

Example:
```python
# src/fraudml/models/gradient_boost.py
from sklearn.ensemble import GradientBoostingClassifier

def train_gb(X_train, y_train, **kwargs):
    model = GradientBoostingClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def predict_gb(model, X):
    return model.predict_proba(X)[:, 1]
```

## Adding New Features

1. Add feature engineering function in `src/fraudml/data/features.py`
2. Update preprocessing in `src/fraudml/data/preprocess.py`
3. Update model metadata in `src/fraudml/models/train.py`
4. Update API schema in `src/fraudml/api/app.py`
5. Update dashboard in `src/fraudml/ui/dashboard.py`
6. Add tests

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
pytest tests/ -v --integration
```

### With Coverage
```bash
pytest tests/ --cov=src/fraudml --cov-report=html
```

## Pull Request Process

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and add tests
3. Run lint and tests: `make lint && make test`
4. Update documentation if needed
5. Create PR with description of changes
6. Get approval from maintainers
7. Merge to main

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance

Example:
```
feat(api): Add batch prediction endpoint

- Added /v1/batch-predict endpoint
- Implemented batch processing logic
- Added unit tests
```

## Release Process

1. Update version in `setup.py` or `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release: `git tag v2.0.0`
5. Push: `git push origin main --tags`
