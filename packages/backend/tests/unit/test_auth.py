"""
Unit tests for authentication module.
Tests JWT token creation, validation, and password hashing.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import hashlib

from ...core.auth.hashing import hash_password, verify_password
from ...core.auth.token import create_access_token, create_refresh_token, verify_token, decode_token
from ...core.auth.service import AuthService
from ...core.auth.models import UserCreate, UserLogin

class TestPasswordHashing:
    """Test password hashing functionality."""
    
    def test_hash_password_success(self):
        """Test successful password hashing."""
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are typically 60 characters
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_hash_password_empty(self):
        """Test hashing empty password raises error."""
        with pytest.raises(ValueError, match="Password cannot be empty"):
            hash_password("")

    def test_verify_password_success(self):
        """Test successful password verification."""
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_failure(self):
        """Test password verification with wrong password."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_empty_inputs(self):
        """Test password verification with empty inputs."""
        assert verify_password("", "hash") is False
        assert verify_password("password", "") is False

class TestJWTTokens:
    """Test JWT token functionality."""
    
    def test_create_access_token_success(self):
        """Test successful access token creation."""
        data = {"sub": "user123"}
        fp_hash = "test_fingerprint_hash"
        
        token = create_access_token(data, fp_hash)
        
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts

    def test_create_access_token_missing_sub(self):
        """Test access token creation fails without sub."""
        data = {"email": "test@example.com"}
        fp_hash = "test_fingerprint_hash"
        
        with pytest.raises(ValueError, match="Token data must include 'sub'"):
            create_access_token(data, fp_hash)

    def test_create_access_token_missing_fingerprint(self):
        """Test access token creation fails without fingerprint."""
        data = {"sub": "user123"}
        
        with pytest.raises(ValueError, match="Device fingerprint hash is required"):
            create_access_token(data, "")

    def test_decode_token_success(self):
        """Test successful token decoding."""
        data = {"sub": "user123"}
        fp_hash = "test_fingerprint_hash"
        
        token = create_access_token(data, fp_hash)
        decoded = decode_token(token)
        
        assert decoded is not None
        assert decoded["sub"] == "user123"
        assert decoded["fp"] == fp_hash
        assert decoded["type"] == "access"

    def test_verify_token_success(self):
        """Test successful token verification."""
        data = {"sub": "user123"}
        fp_hash = "test_fingerprint_hash"
        
        token = create_access_token(data, fp_hash)
        
        assert verify_token(token, fp_hash) is True

    def test_verify_token_wrong_fingerprint(self):
        """Test token verification fails with wrong fingerprint."""
        data = {"sub": "user123"}
        fp_hash = "test_fingerprint_hash"
        wrong_fp_hash = "wrong_fingerprint_hash"
        
        token = create_access_token(data, fp_hash)
        
        assert verify_token(token, wrong_fp_hash) is False

    def test_create_refresh_token_success(self):
        """Test successful refresh token creation."""
        user_id = "user123"
        session_id = "session456"
        fp_hash = "test_fingerprint_hash"
        
        token = create_refresh_token(user_id, session_id, fp_hash)
        decoded = decode_token(token)
        
        assert decoded is not None
        assert decoded["sub"] == user_id
        assert decoded["sid"] == session_id
        assert decoded["fp"] == fp_hash
        assert decoded["type"] == "refresh"

@pytest.mark.asyncio
class TestAuthService:
    """Test authentication service."""
    
    @patch('packages.backend.core.auth.service.create_client')
    async def test_register_user_success(self, mock_create_client):
        """Test successful user registration."""
        # Mock Supabase client
        mock_supabase = Mock()
        mock_create_client.return_value = mock_supabase
        
        # Mock user doesn't exist
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        
        # Mock successful user creation
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [{
            "id": "user123",
            "email": "test@example.com"
        }]
        
        # Mock session creation
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()
        
        auth_service = AuthService()
        user_data = UserCreate(
            email="test@example.com",
            password="password123",
            fp="device_fingerprint"
        )
        
        result = await auth_service.register_user(user_data)
        
        assert result.access_token is not None
        assert result.refresh_token is not None
        assert result.trust_score == 50

    @patch('packages.backend.core.auth.service.create_client')
    async def test_register_user_email_exists(self, mock_create_client):
        """Test user registration fails when email exists."""
        # Mock Supabase client
        mock_supabase = Mock()
        mock_create_client.return_value = mock_supabase
        
        # Mock user exists
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{"id": "existing"}]
        
        auth_service = AuthService()
        user_data = UserCreate(
            email="test@example.com",
            password="password123",
            fp="device_fingerprint"
        )
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.register_user(user_data)
        
        assert exc_info.value.status_code == 400
        assert "already registered" in exc_info.value.detail

if __name__ == "__main__":
    pytest.main([__file__])
    