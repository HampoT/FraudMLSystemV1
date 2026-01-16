import os
import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy.pool import QueuePool
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import contextmanager


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/fraud_db")

Base = declarative_base()

# Connection pool configuration
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
POOL_MAX_OVERFLOW = int(os.getenv("DB_POOL_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 minutes

# Singleton engine with connection pooling
_engine = None


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
    """Get or create the database engine with connection pooling."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=POOL_SIZE,
            max_overflow=POOL_MAX_OVERFLOW,
            pool_timeout=POOL_TIMEOUT,
            pool_recycle=POOL_RECYCLE,
            pool_pre_ping=True,  # Check connection health before use
        )
    return _engine


# Create scoped session factory (lazy init)
_session_factory = None


def _get_session_factory():
    """Get or create the scoped session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = scoped_session(sessionmaker(bind=get_engine()))
    return _session_factory


@contextmanager
def get_session():
    """Context manager for database sessions with automatic cleanup."""
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def store_transaction(data: Dict) -> str:
    """Store transaction record."""
    with get_session() as session:
        tx = TransactionModel(**data)
        session.add(tx)
        return data.get("transaction_id")


def store_prediction(prediction_data: Dict) -> str:
    """Store prediction log."""
    with get_session() as session:
        log = PredictionLog(**prediction_data)
        session.add(log)
        return prediction_data.get("prediction_id")


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


def get_pool_status() -> Dict:
    """Get connection pool statistics."""
    engine = get_engine()
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalidatedcount() if hasattr(pool, 'invalidatedcount') else 0
    }


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully")
    print(f"Pool status: {get_pool_status()}")
