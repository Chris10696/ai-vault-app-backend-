"""
JWT token management for secure authentication.
Handles access and refresh token creation, validation, and decoding.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError
import logging

logger = logging.getLogger(__name__)

class TokenConfig:
    """JWT token configuration."""
    SECRET_KEY = os.getenv("JWT_SECRET", "fallback-dev-secret-change-in-production")
    ALGORITHM = "HS512"  # More secure than HS256
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TTL", 15))  # 15 minutes
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TTL", 7))    # 7 days

def create_access_token(data: Dict[str, Any], fp_hash: str) -> str:
    """
    Create a JWT access token with user data and device fingerprint.
    
    Args:
        data: Payload data to include in token (user_id, etc.)
        fp_hash: SHA-256 hash of device fingerprint
        
    Returns:
        Encoded JWT token string
        
    Raises:
        ValueError: If required data is missing
    """
    if not data.get("sub"):
        raise ValueError("Token data must include 'sub' (subject)")
    
    if not fp_hash:
        raise ValueError("Device fingerprint hash is required")
    
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=TokenConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "fp": fp_hash,
        "type": "access"
    })
    
    try:
        encoded_jwt = jwt.encode(to_encode, TokenConfig.SECRET_KEY, algorithm=TokenConfig.ALGORITHM)
        logger.debug(f"Access token created for user: {data.get('sub')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise

def create_refresh_token(user_id: str, session_id: str, fp_hash: str) -> str:
    """
    Create a JWT refresh token bound to user session and device.
    
    Args:
        user_id: User identifier
        session_id: Session identifier  
        fp_hash: SHA-256 hash of device fingerprint
        
    Returns:
        Encoded JWT refresh token
    """
    to_encode = {
        "sub": user_id,
        "sid": session_id,
        "fp": fp_hash,
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=TokenConfig.REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow()
    }
    
    try:
        encoded_jwt = jwt.encode(to_encode, TokenConfig.SECRET_KEY, algorithm=TokenConfig.ALGORITHM)
        logger.debug(f"Refresh token created for user: {user_id}, session: {session_id}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create refresh token: {e}")
        raise

def verify_token(token: str, expected_fp_hash: str) -> bool:
    """
    Verify token signature and device fingerprint binding.
    
    Args:
        token: JWT token to verify
        expected_fp_hash: Expected device fingerprint hash
        
    Returns:
        True if token is valid and fingerprint matches
    """
    try:
        payload = jwt.decode(token, TokenConfig.SECRET_KEY, algorithms=[TokenConfig.ALGORITHM])
        
        # Check if token is expired
        if datetime.utcnow() > datetime.fromtimestamp(payload.get("exp", 0)):
            logger.warning("Token has expired")
            return False
            
        # Verify device fingerprint binding
        token_fp = payload.get("fp")
        if token_fp != expected_fp_hash:
            logger.warning("Device fingerprint mismatch")
            return False
            
        return True
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return False

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode JWT token and return payload.
    
    Args:
        token: JWT token to decode
        
    Returns:
        Token payload dict or None if invalid
    """
    try:
        payload = jwt.decode(token, TokenConfig.SECRET_KEY, algorithms=[TokenConfig.ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Token decode failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Token decode error: {e}")
        return None
