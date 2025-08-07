"""
Authentication service implementing JWT-based authentication with device fingerprinting.
Handles user registration, login, token refresh, and session management.
"""

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple
from fastapi import HTTPException, status
from supabase import create_client, Client
import logging
import os

from .models import UserCreate, UserLogin, TokenResponse, User, UserSession
from .hashing import hash_password, verify_password
from .token import create_access_token, create_refresh_token, decode_token

logger = logging.getLogger(__name__)

class AuthService:
    """Authentication service handling user management and JWT tokens."""
    
    def __init__(self):
        """Initialize authentication service with Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase configuration missing")
            
        self.supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("AuthService initialized")

    def _hash_fingerprint(self, fingerprint: str) -> str:
        """Create SHA-256 hash of device fingerprint for privacy."""
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    async def register_user(self, user_data: UserCreate) -> TokenResponse:
        """
        Register a new user and return authentication tokens.
        
        Args:
            user_data: User registration data
            
        Returns:
            Token response with access and refresh tokens
            
        Raises:
            HTTPException: If email already exists or registration fails
        """
        try:
            # Check if user already exists
            existing_user = self.supabase.table("users").select("id").eq("email", user_data.email).execute()
            
            if existing_user.data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash password and fingerprint
            pwd_hash = hash_password(user_data.password)
            fp_hash = self._hash_fingerprint(user_data.fp)
            
            # Create user record
            user_result = self.supabase.table("users").insert({
                "id": str(uuid.uuid4()),
                "email": user_data.email,
                "pwd_hash": pwd_hash,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            if not user_result.data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user"
                )
            
            user = user_result.data[0]
            
            # Create session and tokens
            session_id = str(uuid.uuid4())
            access_token = create_access_token({"sub": user["id"]}, fp_hash)
            refresh_token = create_refresh_token(user["id"], session_id, fp_hash)
            
            # Store session
            expires_at = datetime.utcnow() + timedelta(days=7)
            self.supabase.table("user_sessions").insert({
                "session_id": session_id,
                "user_id": user["id"],
                "fp_hash": fp_hash,
                "refresh_sig": hashlib.sha256(refresh_token.encode()).hexdigest()[:64],
                "expires_at": expires_at.isoformat(),
                "last_seen": datetime.utcnow().isoformat()
            }).execute()
            
            logger.info(f"User registered successfully: {user['email']}")
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=900,  # 15 minutes
                trust_score=50   # Base trust score for new users
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )

    async def login_user(self, login_data: UserLogin) -> TokenResponse:
        """
        Authenticate user and return tokens.
        
        Args:
            login_data: User login credentials
            
        Returns:
            Token response with access and refresh tokens
            
        Raises:
            HTTPException: If credentials are invalid
        """
        try:
            # Get user by email
            user_result = self.supabase.table("users").select("*").eq("email", login_data.email).execute()
            
            if not user_result.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            user = user_result.data[0]
            
            # Verify password
            if not verify_password(login_data.password, user["pwd_hash"]):
                logger.warning(f"Failed login attempt for: {login_data.email}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Hash fingerprint
            fp_hash = self._hash_fingerprint(login_data.fp)
            
            # Create new session
            session_id = str(uuid.uuid4())
            access_token = create_access_token({"sub": user["id"]}, fp_hash)
            refresh_token = create_refresh_token(user["id"], session_id, fp_hash)
            
            # Store session
            expires_at = datetime.utcnow() + timedelta(days=7)
            self.supabase.table("user_sessions").insert({
                "session_id": session_id,
                "user_id": user["id"],
                "fp_hash": fp_hash,
                "refresh_sig": hashlib.sha256(refresh_token.encode()).hexdigest()[:64],
                "expires_at": expires_at.isoformat(),
                "last_seen": datetime.utcnow().isoformat()
            }).execute()
            
            # Update last login
            self.supabase.table("users").update({
                "last_login": datetime.utcnow().isoformat()
            }).eq("id", user["id"]).execute()
            
            logger.info(f"User logged in successfully: {user['email']}")
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=900,
                trust_score=60   # Higher trust for returning users
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Login failed"
            )

    async def refresh_token(self, refresh_token: str, fp: str) -> TokenResponse:
        """
        Refresh access token using valid refresh token.
        
        Args:
            refresh_token: Valid refresh token
            fp: Device fingerprint
            
        Returns:
            New token response
            
        Raises:
            HTTPException: If refresh token is invalid
        """
        try:
            # Decode refresh token
            payload = decode_token(refresh_token)
            if not payload or payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            fp_hash = self._hash_fingerprint(fp)
            
            # Verify fingerprint matches
            if payload.get("fp") != fp_hash:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Device fingerprint mismatch"
                )
            
            # Verify session exists and is valid
            session_result = self.supabase.table("user_sessions").select("*").eq(
                "session_id", payload.get("sid")
            ).eq("user_id", payload.get("sub")).execute()
            
            if not session_result.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session not found"
                )
            
            session = session_result.data[0]
            
            # Check session expiry
            if datetime.fromisoformat(session["expires_at"]) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired"
                )
            
            # Create new access token
            access_token = create_access_token({"sub": payload.get("sub")}, fp_hash)
            
            # Update session last seen
            self.supabase.table("user_sessions").update({
                "last_seen": datetime.utcnow().isoformat()
            }).eq("session_id", payload.get("sid")).execute()
            
            logger.info(f"Token refreshed for user: {payload.get('sub')}")
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,  # Keep same refresh token
                expires_in=900,
                trust_score=70  # Higher trust for token refresh
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token refresh failed"
            )

# Global auth service instance
auth_service = AuthService()
