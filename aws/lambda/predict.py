"""
AWS Lambda handler for fraud prediction.

This module provides the Lambda function handler for single predictions.
"""
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Global model reference (loaded once per container)
MODEL = None
META = None


def load_model():
    """Load model from Lambda layer or S3."""
    global MODEL, META
    
    if MODEL is not None:
        return MODEL, META
    
    # Try to load from layer first
    layer_path = "/opt/python/artifacts"
    model_path = os.path.join(layer_path, "model.joblib")
    meta_path = os.path.join(layer_path, "model_meta.json")
    
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
        with open(meta_path, 'r') as f:
            META = json.load(f)
        logger.info("Loaded model from Lambda layer")
    else:
        # Fall back to S3
        import boto3
        s3 = boto3.client('s3')
        bucket = os.getenv("MODEL_BUCKET")
        
        # Download to /tmp
        s3.download_file(bucket, "model.joblib", "/tmp/model.joblib")
        s3.download_file(bucket, "model_meta.json", "/tmp/model_meta.json")
        
        MODEL = joblib.load("/tmp/model.joblib")
        with open("/tmp/model_meta.json", 'r') as f:
            META = json.load(f)
        logger.info("Loaded model from S3")
    
    return MODEL, META


def engineer_features(data: Dict) -> pd.DataFrame:
    """Apply feature engineering to transaction data."""
    df = pd.DataFrame([data])
    
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_zscore'] = 0  # Single sample, can't compute
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)
    df['is_weekend'] = 0
    df['high_amount'] = (df['amount'] > 500).astype(int)
    df['very_high_amount'] = (df['amount'] > 2000).astype(int)
    df['low_device'] = (df['device_score'] < 0.3).astype(int)
    df['medium_device'] = ((df['device_score'] >= 0.3) & (df['device_score'] < 0.7)).astype(int)
    df['high_risk_country'] = (df['country_risk'] >= 4).astype(int)
    df['risk_score'] = (
        df['high_amount'] * 0.3 +
        df['low_device'] * 0.3 +
        df['high_risk_country'] * 0.2 +
        df['is_night'] * 0.2
    )
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['amount_device_interaction'] = df['amount'] * (1 - df['device_score'])
    df['amount_country_interaction'] = df['amount'] * df['country_risk'] / 5
    
    return df


def validate_request(body: Dict) -> tuple:
    """Validate request body."""
    required = ['amount', 'hour', 'device_score', 'country_risk']
    missing = [f for f in required if f not in body]
    
    if missing:
        return False, f"Missing required fields: {missing}"
    
    if body['amount'] < 0:
        return False, "Amount must be non-negative"
    
    if not 0 <= body['hour'] <= 23:
        return False, "Hour must be between 0 and 23"
    
    if not 0 <= body['device_score'] <= 1:
        return False, "Device score must be between 0 and 1"
    
    if not 1 <= body['country_risk'] <= 5:
        return False, "Country risk must be between 1 and 5"
    
    return True, None


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for fraud prediction.
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    import time
    import uuid
    
    start_time = time.time()
    
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        # Validate API key
        headers = event.get('headers', {})
        api_key = headers.get('x-api-key') or headers.get('X-API-Key')
        
        if api_key != os.getenv('API_KEY'):
            return {
                'statusCode': 401,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid API key'})
            }
        
        # Validate request
        valid, error = validate_request(body)
        if not valid:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': error})
            }
        
        # Load model
        model, meta = load_model()
        
        # Engineer features
        df = engineer_features(body)
        
        # Get feature columns
        feature_cols = [
            'amount', 'hour', 'device_score', 'country_risk',
            'amount_log', 'amount_zscore',
            'is_night', 'is_evening', 'is_weekend',
            'high_amount', 'very_high_amount',
            'low_device', 'medium_device',
            'high_risk_country',
            'risk_score',
            'hour_sin', 'hour_cos',
            'amount_device_interaction', 'amount_country_interaction'
        ]
        
        features = df[feature_cols]
        
        # Make prediction
        threshold = meta.get('threshold', 0.5)
        prob = float(model.predict_proba(features)[0, 1])
        label = int(prob >= threshold)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            'fraud_probability': round(prob, 6),
            'fraud_label': label,
            'prediction_id': str(uuid.uuid4()),
            'model_version': meta.get('model_version', 'unknown'),
            'model_type': meta.get('model_type', 'unknown'),
            'threshold': threshold,
            'processing_time_ms': round(processing_time, 2),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'X-Request-Id': response['prediction_id']
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
