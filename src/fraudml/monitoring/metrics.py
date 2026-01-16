from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST


PREDICTION_COUNTER = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions',
    ['model_type', 'fraud_label']
)

FRAUD_POSITIVE_PREDICTIONS = Counter(
    'fraud_positive_predictions_total',
    'Total number of fraud predictions (fraud_label=1)',
    ['model_type']
)

PREDICTION_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_type'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_SUMMARY = Summary(
    'fraud_prediction_latency_summary_seconds',
    'Prediction latency summary in seconds',
    ['model_type']
)

FRAUD_RATE = Gauge(
    'fraud_detection_rate',
    'Current fraud detection rate (rolling 1h)',
    ['model_type']
)

MODEL_VERSION = Gauge(
    'fraud_model_version',
    'Current model version',
    ['model_type']
)

API_REQUESTS_TOTAL = Counter(
    'fraud_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

API_REQUESTS_IN_PROGRESS = Gauge(
    'fraud_api_requests_in_progress',
    'Number of API requests in progress',
    ['endpoint']
)


def record_prediction(model_type: str, fraud_label: int, latency: float):
    """Record a prediction for metrics."""
    PREDICTION_COUNTER.labels(model_type=model_type, fraud_label=str(fraud_label)).inc()
    if fraud_label == 1:
        FRAUD_POSITIVE_PREDICTIONS.labels(model_type=model_type).inc()
    PREDICTION_LATENCY.labels(model_type=model_type).observe(latency)
    PREDICTION_SUMMARY.labels(model_type=model_type).observe(latency)


def set_fraud_rate(model_type: str, rate: float):
    """Update the fraud rate gauge."""
    FRAUD_RATE.labels(model_type=model_type).set(rate)


def set_model_version(model_type: str, version: str):
    """Update the model version gauge."""
    MODEL_VERSION.labels(model_type=model_type).set(float(hash(version) % 1000000))


def record_api_request(endpoint: str, method: str, status: int):
    """Record an API request."""
    API_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=str(status)).inc()


def increment_in_progress(endpoint: str):
    """Increment in-progress requests gauge."""
    API_REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()


def decrement_in_progress(endpoint: str):
    """Decrement in-progress requests gauge."""
    API_REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


def get_metrics():
    """Generate Prometheus metrics output."""
    return generate_latest()


def get_content_type():
    """Get content type for Prometheus response."""
    return CONTENT_TYPE_LATEST
