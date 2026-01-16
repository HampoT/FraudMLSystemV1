"""
JWT Authentication module for Fraud Detection API.

Provides JWT token generation, validation, and role-based access control.
Supports both JWT and API key authentication for backward compatibility.
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from functools import wraps

from fastapi import HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Try to import jose for JWT handling
try:
    from jose import jwt, JWTError
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    JWTError = Exception  # Fallback

logger = logging.getLogger(__name__)

# Configuration from environment
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
API_KEY = os.getenv("API_KEY", "default-api-key")

# Role definitions
ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLE_SERVICE = "service"

ROLE_PERMISSIONS = {
    ROLE_ADMIN: ["predict", "batch_predict", "explain", "admin", "metrics"],
    ROLE_USER: ["predict", "batch_predict", "explain"],
    ROLE_SERVICE: ["predict", "batch_predict", "metrics"],
}


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str  # Subject (user ID)
    role: str = ROLE_USER
    exp: Optional[int] = None
    iat: Optional[int] = None
    permissions: Optional[List[str]] = None


class TokenResponse(BaseModel):
    """Response model for token generation."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    role: str


class AuthenticatedUser(BaseModel):
    """Authenticated user context."""
    user_id: str
    role: str
    permissions: List[str]
    auth_method: str  # "jwt" or "api_key"


# Security scheme
security = HTTPBearer(auto_error=False)


def create_access_token(
    user_id: str,
    role: str = ROLE_USER,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a new JWT access token.
    
    Args:
        user_id: Unique user identifier
        role: User role (admin, user, service)
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="JWT support not available. Install python-jose."
        )
    
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRY_HOURS)
    
    expire = datetime.utcnow() + expires_delta
    permissions = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS[ROLE_USER])
    
    payload = {
        "sub": user_id,
        "role": role,
        "permissions": permissions,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    logger.info(f"Created token for user {user_id} with role {role}")
    return token


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token.
    
    Args:
        token: Encoded JWT token string
        
    Returns:
        TokenPayload with decoded claims
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="JWT support not available. Install python-jose."
        )
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(
            sub=payload.get("sub"),
            role=payload.get("role", ROLE_USER),
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            permissions=payload.get("permissions", [])
        )
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )


def verify_api_key(api_key: str) -> bool:
    """Verify an API key.
    
    Args:
        api_key: The API key to verify
        
    Returns:
        True if valid, False otherwise
    """
    return api_key == API_KEY


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_api_key: Optional[str] = Header(None)
) -> AuthenticatedUser:
    """Get the current authenticated user from JWT or API key.
    
    Supports dual authentication:
    1. Bearer token (JWT) in Authorization header
    2. API key in X-API-Key header
    
    Args:
        request: FastAPI request object
        credentials: Optional Bearer token
        x_api_key: Optional API key header
        
    Returns:
        AuthenticatedUser with user context
        
    Raises:
        HTTPException: If neither auth method is valid
    """
    # Try JWT authentication first
    if credentials and credentials.credentials:
        token = credentials.credentials
        try:
            payload = decode_token(token)
            return AuthenticatedUser(
                user_id=payload.sub,
                role=payload.role,
                permissions=payload.permissions or ROLE_PERMISSIONS.get(payload.role, []),
                auth_method="jwt"
            )
        except HTTPException:
            pass  # Fall through to API key auth
    
    # Try API key authentication
    if x_api_key and verify_api_key(x_api_key):
        return AuthenticatedUser(
            user_id="api_key_user",
            role=ROLE_SERVICE,
            permissions=ROLE_PERMISSIONS[ROLE_SERVICE],
            auth_method="api_key"
        )
    
    # No valid authentication
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permission(permission: str):
    """Decorator to require a specific permission.
    
    Args:
        permission: Required permission string
        
    Returns:
        Dependency that checks for the permission
    """
    async def permission_checker(
        user: AuthenticatedUser = Depends(get_current_user)
    ) -> AuthenticatedUser:
        if permission not in user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return user
    
    return permission_checker


def require_role(role: str):
    """Decorator to require a specific role.
    
    Args:
        role: Required role string
        
    Returns:
        Dependency that checks for the role
    """
    async def role_checker(
        user: AuthenticatedUser = Depends(get_current_user)
    ) -> AuthenticatedUser:
        if user.role != role and user.role != ROLE_ADMIN:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' required"
            )
        return user
    
    return role_checker


# Rate limiting per user helper
def get_user_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on authenticated user.
    
    Falls back to IP address if not authenticated.
    """
    # Try to get user from request state (set by auth middleware)
    user = getattr(request.state, "user", None)
    if user and isinstance(user, AuthenticatedUser):
        return f"user:{user.user_id}"
    
    # Fall back to IP-based limiting
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host}"
