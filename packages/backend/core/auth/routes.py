"""
FastAPI routes for authentication endpoints.
Handles user registration, login, and token refresh.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

from .models import UserCreate, UserLogin, TokenResponse, TokenRefresh
from .service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate) -> TokenResponse:
    """
    Register a new user account.
    
    Args:
        user_data: User registration data including email, password, and device fingerprint
        
    Returns:
        JWT tokens and user session data
        
    Raises:
        HTTPException: 400 if email exists, 500 if registration fails
    """
    try:
        result = await auth_service.register_user(user_data)
        logger.info(f"New user registered: {user_data.email}")
        return result
    except Exception as e:
        logger.error(f"Signup failed for {user_data.email}: {e}")
        raise

@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin) -> TokenResponse:
    """
    Authenticate user and return tokens.
    
    Args:
        login_data: User login credentials and device fingerprint
        
    Returns:
        JWT tokens and session data
        
    Raises:
        HTTPException: 401 if credentials are invalid
    """
    try:
        result = await auth_service.login_user(login_data)
        logger.info(f"User logged in: {login_data.email}")
        return result
    except Exception as e:
        logger.error(f"Login failed for {login_data.email}: {e}")
        raise

@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    token_data: TokenRefresh,
    x_device_fingerprint: Optional[str] = Header(None, alias="X-Device-Fingerprint")
) -> TokenResponse:
    """
    Refresh access token using valid refresh token.
    
    Args:
        token_data: Refresh token
        x_device_fingerprint: Device fingerprint header
        
    Returns:
        New access token and existing refresh token
        
    Raises:
        HTTPException: 401 if refresh token is invalid or fingerprint mismatch
    """
    if not x_device_fingerprint:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Device fingerprint header required"
        )
    
    try:
        result = await auth_service.refresh_token(token_data.refresh_token, x_device_fingerprint)
        logger.info("Access token refreshed successfully")
        return result
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise

@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_device_fingerprint: Optional[str] = Header(None, alias="X-Device-Fingerprint")
):
    """
    Logout user and invalidate session.
    
    Args:
        credentials: Bearer token
        x_device_fingerprint: Device fingerprint header
        
    Returns:
        Success message
    """
    # TODO: Implement session invalidation in Sprint A2
    # For now, just return success as tokens will expire naturally
    logger.info("User logged out")
    return {"message": "Logged out successfully"}

@router.get("/health")
async def health_check():
    """Health check endpoint for authentication service."""
    return {"status": "healthy", "service": "auth"}
