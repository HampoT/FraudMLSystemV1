"""
Audit logging module for Fraud Detection API.

Provides comprehensive audit logging for all predictions and admin actions.
Supports async logging to avoid impacting request latency.
"""
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from queue import Queue
from threading import Thread
from dataclasses import dataclass, asdict

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/fraud_db")

Base = declarative_base()


class AuditLog(Base):
    """Audit log database model."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    auth_method = Column(String(20))  # jwt, api_key
    ip_address = Column(String(50))
    endpoint = Column(String(100))
    method = Column(String(10))
    request_id = Column(String(50), index=True)
    prediction_id = Column(String(50), index=True)
    transaction_amount = Column(Float)
    fraud_probability = Column(Float)
    fraud_label = Column(Integer)
    processing_time_ms = Column(Float)
    status_code = Column(Integer)
    error_message = Column(Text)
    metadata = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_type: str
    user_id: str
    auth_method: str = "unknown"
    ip_address: str = ""
    endpoint: str = ""
    method: str = ""
    request_id: str = ""
    prediction_id: str = ""
    transaction_amount: float = None
    fraud_probability: float = None
    fraud_label: int = None
    processing_time_ms: float = None
    status_code: int = 200
    error_message: str = None
    metadata: Dict = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class AuditLogger:
    """Async audit logger with background processing."""

    def __init__(self, max_queue_size: int = 1000, batch_size: int = 50):
        """Initialize the audit logger.
        
        Args:
            max_queue_size: Maximum events to queue before blocking
            batch_size: Number of events to process in each batch
        """
        self.queue = Queue(maxsize=max_queue_size)
        self.batch_size = batch_size
        self._engine = None
        self._session_factory = None
        self._worker_thread = None
        self._running = False

    @property
    def engine(self):
        """Lazy initialization of database engine."""
        if self._engine is None:
            self._engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=2,
                max_overflow=3,
                pool_pre_ping=True
            )
        return self._engine

    @property
    def session_factory(self):
        """Lazy initialization of session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    def init_db(self):
        """Initialize audit log database table."""
        Base.metadata.create_all(self.engine)
        logger.info("Audit log table initialized")

    def start(self):
        """Start the background worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        logger.info("Audit logger worker started")

    def stop(self):
        """Stop the background worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Audit logger worker stopped")

    def log(self, event: AuditEvent):
        """Queue an audit event for processing.
        
        This is non-blocking unless the queue is full.
        
        Args:
            event: Audit event to log
        """
        try:
            self.queue.put_nowait(event)
        except Exception as e:
            logger.warning(f"Failed to queue audit event: {e}")

    def log_prediction(
        self,
        user_id: str,
        auth_method: str,
        ip_address: str,
        request_id: str,
        prediction_id: str,
        transaction_amount: float,
        fraud_probability: float,
        fraud_label: int,
        processing_time_ms: float,
        endpoint: str = "/v1/predict"
    ):
        """Log a prediction event.
        
        Args:
            user_id: Authenticated user ID
            auth_method: Authentication method used
            ip_address: Client IP address
            request_id: Unique request identifier
            prediction_id: Unique prediction identifier
            transaction_amount: Transaction amount
            fraud_probability: Predicted fraud probability
            fraud_label: Predicted fraud label (0/1)
            processing_time_ms: Processing time in milliseconds
            endpoint: API endpoint called
        """
        event = AuditEvent(
            event_type="prediction",
            user_id=user_id,
            auth_method=auth_method,
            ip_address=ip_address,
            endpoint=endpoint,
            method="POST",
            request_id=request_id,
            prediction_id=prediction_id,
            transaction_amount=transaction_amount,
            fraud_probability=fraud_probability,
            fraud_label=fraud_label,
            processing_time_ms=processing_time_ms,
            status_code=200
        )
        self.log(event)

    def log_error(
        self,
        user_id: str,
        ip_address: str,
        endpoint: str,
        method: str,
        status_code: int,
        error_message: str,
        request_id: str = ""
    ):
        """Log an error event.
        
        Args:
            user_id: Authenticated user ID
            ip_address: Client IP address
            endpoint: API endpoint called
            method: HTTP method
            status_code: HTTP status code
            error_message: Error message
            request_id: Unique request identifier
        """
        event = AuditEvent(
            event_type="error",
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            request_id=request_id,
            status_code=status_code,
            error_message=error_message
        )
        self.log(event)

    def log_admin_action(
        self,
        user_id: str,
        action: str,
        details: Dict = None
    ):
        """Log an admin action.
        
        Args:
            user_id: Admin user ID
            action: Action performed
            details: Additional action details
        """
        event = AuditEvent(
            event_type="admin_action",
            user_id=user_id,
            metadata={"action": action, **(details or {})}
        )
        self.log(event)

    def _process_queue(self):
        """Background worker to process queued events."""
        while self._running:
            batch = []
            
            # Collect batch of events
            while len(batch) < self.batch_size:
                try:
                    event = self.queue.get(timeout=1)
                    batch.append(event)
                except Exception:
                    break  # Timeout or queue empty
            
            if batch:
                self._persist_batch(batch)

    def _persist_batch(self, events: list):
        """Persist a batch of events to database.
        
        Args:
            events: List of AuditEvent objects
        """
        session = self.session_factory()
        try:
            for event in events:
                log_entry = AuditLog(
                    event_type=event.event_type,
                    user_id=event.user_id,
                    auth_method=event.auth_method,
                    ip_address=event.ip_address,
                    endpoint=event.endpoint,
                    method=event.method,
                    request_id=event.request_id,
                    prediction_id=event.prediction_id,
                    transaction_amount=event.transaction_amount,
                    fraud_probability=event.fraud_probability,
                    fraud_label=event.fraud_label,
                    processing_time_ms=event.processing_time_ms,
                    status_code=event.status_code,
                    error_message=event.error_message,
                    metadata=json.dumps(event.metadata) if event.metadata else None,
                    created_at=event.timestamp
                )
                session.add(log_entry)
            
            session.commit()
            logger.debug(f"Persisted {len(events)} audit events")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to persist audit events: {e}")
        finally:
            session.close()


# Global audit logger instance
audit_logger = AuditLogger()


def start_audit_logger():
    """Initialize and start the audit logger."""
    try:
        audit_logger.init_db()
        audit_logger.start()
    except Exception as e:
        logger.warning(f"Audit logger initialization failed: {e}")


def stop_audit_logger():
    """Stop the audit logger."""
    audit_logger.stop()
