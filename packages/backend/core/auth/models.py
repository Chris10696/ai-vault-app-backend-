"""
Pydantic models for authentication and user management.
Defines data structures for API requests and database entities.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
import uuid

class UserCreate(BaseModel):
    """Model for user registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    fp: str = Field(..., description="Device fingerprint hash")

class UserLogin(BaseModel):
    """Model for user login request."""
    email: EmailStr
    password: str
    fp: str = Field(..., description="Device fingerprint hash")

class TokenResponse(BaseModel):
    """Model for token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    trust_score: Optional[int] = None

class TokenRefresh(BaseModel):
    """Model for token refresh request."""
    refresh_token: str

class TokenData(BaseModel):
    """Model for decoded token data."""
    sub: str  # user_id
    fp: str   # fingerprint hash
    exp: int  # expiration timestamp
    iat: int  # issued at timestamp
    type: str # token type (access/refresh)

class User(BaseModel):
    """User data model."""
    id: str
    email: str
    created_at: datetime
    last_login: Optional[datetime] = None

class UserSession(BaseModel):
    """User session model."""
    session_id: str
    user_id: str
    fp_hash: str
    refresh_token_signature: str
    expires_at: datetime
    last_seen: datetime
    created_at: datetime
