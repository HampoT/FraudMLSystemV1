import os
import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Optional, List, Dict
from datetime import datetime


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/fraud_db")

Base = declarative_base()


class TransactionModel(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), unique=True, nullable=False)
    user_id = Column(String(50), nullable=False)
    merchant_id = Column(String(50))
    amount = Column(Float, nullable=False)
    hour = Column(Integer, nullable=False)
    device_score = Column(Float, nullable=False)
    country_risk = Column(Integer, nullable=False)
    fraud_probability = Column(Float)
    fraud_label = Column(Integer)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(50), unique=True, nullable=False)
    transaction_id = Column(String(50))
    model_type = Column(String(50))
    fraud_probability = Column(Float)
    fraud_label = Column(Integer)
    threshold = Column(Float)
    processing_time_ms = Column(Float)
    features = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def get_session():
    Session = sessionmaker(bind=get_engine())
    return Session()


def init_database():
    engine = get_engine()
    Base.metadata.create_all(engine)


def store_transaction(data: Dict) -> str:
    """Store transaction record."""
    session = get_session()
    try:
        tx = TransactionModel(**data)
        session.add(tx)
        session.commit()
        return data.get("transaction_id")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def store_prediction(prediction_data: Dict) -> str:
    """Store prediction log."""
    session = get_session()
    try:
        log = PredictionLog(**prediction_data)
        session.add(log)
        session.commit()
        return prediction_data.get("prediction_id")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_transactions(limit: int = 100) -> pd.DataFrame:
    """Retrieve transactions from database."""
    query = text("SELECT * FROM transactions ORDER BY created_at DESC LIMIT :limit")
    engine = get_engine()
    return pd.read_sql(query, engine, params={"limit": limit})


def get_predictions(start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
    """Retrieve predictions from database with time filter."""
    if start_time and end_time:
        query = text("""
            SELECT * FROM prediction_logs
            WHERE created_at BETWEEN :start AND :end
            ORDER BY created_at DESC
        """)
        engine = get_engine()
        return pd.read_sql(query, engine, params={"start": start_time, "end": end_time})
    else:
        query = text("SELECT * FROM prediction_logs ORDER BY created_at DESC LIMIT 1000")
        engine = get_engine()
        return pd.read_sql(query, engine)


def get_fraud_rate_by_hour() -> pd.DataFrame:
    """Get fraud rate aggregated by hour."""
    query = text("""
        SELECT
            hour,
            COUNT(*) as total_transactions,
            SUM(CASE WHEN fraud_label = 1 THEN 1 ELSE 0 END) as fraud_count,
            AVG(fraud_probability) as avg_fraud_prob
        FROM transactions
        GROUP BY hour
        ORDER BY hour
    """)
    engine = get_engine()
    return pd.read_sql(query, engine)


def get_model_performance_summary() -> Dict:
    """Get summary of model performance from database."""
    query = text("""
        SELECT
            model_type,
            COUNT(*) as total_predictions,
            AVG(CASE WHEN fraud_label = 1 THEN fraud_probability ELSE NULL END) as avg_prob_fraud,
            AVG(CASE WHEN fraud_label = 0 THEN fraud_probability ELSE NULL END) as avg_prob_legit,
            AVG(processing_time_ms) as avg_processing_time
        FROM prediction_logs
        GROUP BY model_type
    """)
    engine = get_engine()
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully")
