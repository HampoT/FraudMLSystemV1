import os
import json
import hashlib
import secrets
import jwt
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class TokenType(Enum):
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class TokenPayload:
    user_id: str
    token_type: TokenType
    exp: int
    iat: int
    permissions: list[str] = None


class User(BaseModel):
    user_id: str
    username: str
    email: str
    role: str = "user"
    permissions: list[str] = []
    created_at: datetime = None
    last_login: datetime = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class AuthConfig:
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_EXPIRE", 30))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_EXPIRE", 7))
    API_KEY_HEADER: str = "X-API-Key"


class JWTAuth:
    """JWT-based authentication handler."""

    def __init__(self, secret_key: str = None, algorithm: str = None):
        self.secret_key = secret_key or AuthConfig.SECRET_KEY
        self.algorithm = algorithm or AuthConfig.ALGORITHM
        self._api_keys: Dict[str, Dict] = {}
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from environment or file."""
        api_keys_env = os.getenv("API_KEYS", "")
        if api_keys_env:
            try:
                keys = json.loads(api_keys_env)
                self._api_keys = {k: v for k, v in keys.items()}
            except json.JSONDecodeError:
                logger.warning("Failed to parse API_KEYS environment variable")

    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verify API key and return associated user data."""
        if not api_key:
            return None
        
        if api_key in self._api_keys:
            return self._api_keys[api_key]
        
        default_key = os.getenv("API_KEY", "")
        if api_key == default_key:
            return {"user_id": "default", "role": "admin", "permissions": ["*"]}
        
        return None

    def create_access_token(self, user_id: str, permissions: list = None) -> Tuple[str, int]:
        """Create a new access token."""
        expire = int(time.time()) + AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        payload = {
            "user_id": user_id,
            "token_type": TokenType.ACCESS.value,
            "exp": expire,
            "iat": int(time.time()),
            "permissions": permissions or []
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, expire

    def create_refresh_token(self, user_id: str) -> Tuple[str, int]:
        """Create a new refresh token."""
        expire = int(time.time()) + AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        payload = {
            "user_id": user_id,
            "token_type": TokenType.REFRESH.value,
            "exp": expire,
            "iat": int(time.time())
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, expire

    def verify_token(self, token: str) -> TokenPayload:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return TokenPayload(
                user_id=payload["user_id"],
                token_type=TokenType(payload["token_type"]),
                exp=payload["exp"],
                iat=payload["iat"],
                permissions=payload.get("permissions", [])
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> Tuple[str, str, int]:
        """Use refresh token to get new access token."""
        payload = self.verify_token(refresh_token)
        
        if payload.token_type != TokenType.REFRESH:
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        new_access, expires = self.create_access_token(payload.user_id, payload.permissions)
        new_refresh, _ = self.create_refresh_token(payload.user_id)
        
        return new_access, new_refresh, expires

    def extract_token(self, credentials: HTTPAuthorizationCredentials = None,
                      api_key: str = None, user_id: str = None) -> Tuple[str, Optional[str], str]:
        """Extract authentication credentials from request.
        
        Returns: (token_type, token, user_id)
        """
        if api_key:
            return "api_key", api_key, "api_key_user"
        
        if credentials and credentials.scheme == "Bearer":
            return "jwt", credentials.credentials, "jwt_user"
        
        if user_id:
            return "internal", None, user_id
        
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials"
        )


security = HTTPBearer()
auth_handler = JWTAuth()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    api_key: str = None
) -> TokenPayload:
    """Dependency to get current authenticated user from JWT."""
    token_type, token, _ = auth_handler.extract_token(credentials, api_key=api_key)
    
    if token_type == "api_key":
        user_data = auth_handler.verify_api_key(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return TokenPayload(
            user_id=user_data.get("user_id", "unknown"),
            token_type=TokenType.ACCESS,
            exp=int(time.time()) + 3600,
            iat=int(time.time()),
            permissions=user_data.get("permissions", [])
        )
    
    return auth_handler.verify_token(token)


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    api_key: str = None
) -> Optional[TokenPayload]:
    """Get current user if authenticated, None otherwise."""
    try:
        return await get_current_user(credentials, api_key)
    except HTTPException:
        return None


def require_permission(permission: str):
    """Decorator to require specific permission."""
    async def permission_checker(user: TokenPayload = Depends(get_current_user)):
        if "*" not in user.permissions and permission not in user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return user
    return permission_checker


def require_role(role: str):
    """Decorator to require specific role."""
    async def role_checker(user: TokenPayload = Depends(get_current_user)):
        user_data = auth_handler.verify_api_key(
            getattr(user, "api_key", None) or ""
        )
        user_role = user_data.get("role", "user") if user_data else "user"
        
        if user_role != role and user_role != "admin":
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' required"
            )
        return user
    return role_checker


class APIKeyAuth:
    """API Key authentication for service-to-service communication."""

    def __init__(self):
        self._keys: Dict[str, Dict] = {}
        self._load_keys()

    def _load_keys(self):
        """Load API keys from environment."""
        default_key = os.getenv("API_KEY", "default-api-key")
        self._keys[default_key] = {
            "name": "default",
            "permissions": ["predict", "batch_predict"],
            "rate_limit": 100
        }

    def create_api_key(self, name: str, permissions: list, rate_limit: int = 100) -> str:
        """Create a new API key."""
        api_key = f"fk_{secrets.token_hex(16)}"
        self._keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit
        }
        return api_key

    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verify API key and return metadata."""
        if api_key in self._keys:
            return self._keys[api_key]
        return None

    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission."""
        key_data = self.verify_api_key(api_key)
        if not key_data:
            return False
        return "*" in key_data["permissions"] or permission in key_data["permissions"]


api_key_auth = APIKeyAuth()


async def verify_api_authentication(
    api_key: str = None,
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict:
    """Verify authentication using either API key or JWT."""
    token_type, token, _ = auth_handler.extract_token(credentials, api_key=api_key)
    
    if token_type == "api_key":
        user_data = auth_handler.verify_api_key(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {"type": "api_key", "data": user_data}
    
    payload = auth_handler.verify_token(token)
    return {"type": "jwt", "data": payload}


class AuditLogger:
    """Audit logging for all predictions and sensitive operations."""

    def __init__(self):
        self._logs: list = []

    def log(self, action: str, user_id: str, details: Dict, ip_address: str = None):
        """Log an audit event."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user_id": user_id,
            "details": details,
            "ip_address": ip_address
        }
        self._logs.append(log_entry)
        logger.info(f"Audit: {action} by {user_id} - {json.dumps(details)}")

    def get_logs(self, user_id: str = None, action: str = None,
                 start_time: datetime = None, end_time: datetime = None) -> list:
        """Query audit logs with filters."""
        logs = self._logs

        if user_id:
            logs = [l for l in logs if l["user_id"] == user_id]
        if action:
            logs = [l for l in logs if l["action"] == action]
        if start_time:
            logs = [l for l in logs if l["timestamp"] >= start_time.isoformat()]
        if end_time:
            logs = [l for l in logs if l["timestamp"] <= end_time.isoformat()]

        return logs

    def log_prediction(self, user_id: str, transaction: Dict, result: Dict,
                       model_version: str, latency_ms: float, ip_address: str = None):
        """Log a prediction request."""
        self.log(
            action="prediction",
            user_id=user_id,
            details={
                "transaction": transaction,
                "result": result,
                "model_version": model_version,
                "latency_ms": latency_ms
            },
            ip_address=ip_address
        )

    def log_auth_failure(self, attempt_type: str, details: Dict, ip_address: str = None):
        """Log authentication failure."""
        self.log(
            action="auth_failure",
            user_id="anonymous",
            details={
                "attempt_type": attempt_type,
                "details": str(details)
            },
            ip_address=ip_address
        )

    def export_logs(self, format: str = "json") -> str:
        """Export all logs."""
        if format == "json":
            return json.dumps(self._logs, indent=2)
        return str(self._logs)


audit_logger = AuditLogger()
