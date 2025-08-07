"""
Advanced rate limiting middleware with Redis backend.
Implements sliding window rate limiting with different limits per endpoint and user.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import os

logger = logging.getLogger(__name__)

class RateLimitConfig:
    """Rate limiting configuration."""
    
    # Default limits (requests per minute)
    DEFAULT_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", 100))
    
    # Endpoint-specific limits
    ENDPOINT_LIMITS = {
        "/api/auth/login": 10,      # Lower limit for login attempts
        "/api/auth/signup": 5,      # Very low limit for signups
        "/api/auth/refresh": 20,    # Medium limit for token refresh
        "/api/apps/create": 30,     # Limit for app creation
        "/api/memory/*": 200,       # Higher limit for memory operations
    }
    
    # Burst limits (short-term spike protection)
    BURST_LIMITS = {
        "/api/auth/login": (3, 10),    # 3 requests per 10 seconds
        "/api/auth/signup": (2, 60),   # 2 requests per minute
    }

class RateLimitMiddleware:
    """
    Rate limiting middleware using sliding window algorithm with Redis.
    Provides per-IP and per-user rate limiting with different limits per endpoint.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.fallback_store: Dict[str, Dict] = {}  # In-memory fallback
        
    async def __call__(self, request: Request, call_next):
        """
        Apply rate limiting to incoming requests.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response or rate limit error
        """
        # Initialize Redis client if not already done
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(
                    os.getenv("REDIS_URL", "redis://localhost:6379"),
                    encoding="utf8",
                    decode_responses=True
                )
                await self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
        
        # Skip rate limiting for health checks and static files
        if self._should_skip_rate_limit(request.url.path):
            return await call_next(request)
        
        try:
            # Get rate limit key and limits
            limit_key = self._get_rate_limit_key(request)
            limits = self._get_limits_for_path(request.url.path)
            
            # Check rate limits
            allowed, retry_after = await self._check_rate_limit(limit_key, limits)
            
            if not allowed:
                # Log rate limit violation
                await self._log_rate_limit_violation(request, limit_key)
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": retry_after,
                        "limit": limits["per_minute"]
                    },
                    headers={"Retry-After": str(retry_after)}
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            if isinstance(response, Response):
                remaining = await self._get_remaining_requests(limit_key, limits)
                response.headers["X-RateLimit-Limit"] = str(limits["per_minute"])
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(int((datetime.utcnow() + timedelta(minutes=1)).timestamp()))
            
            return response
        
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request to proceed if rate limiting fails
            return await call_next(request)

    def _should_skip_rate_limit(self, path: str) -> bool:
        """Check if path should skip rate limiting."""
        skip_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}
        return path in skip_paths or path.startswith("/static/")

    def _get_rate_limit_key(self, request: Request) -> str:
        """
        Generate rate limit key for request.
        Combines IP address and user ID (if authenticated) for accurate limiting.
        """
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Get user ID from token (if available)
        user_id = None
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from ...core.auth.token import decode_token
                token = auth_header.split(" ")[1]
                payload = decode_token(token)
                if payload:
                    user_id = payload.get("sub")
            except:
                pass
        
        # Create composite key
        if user_id:
            key_base = f"user:{user_id}:{client_ip}"
        else:
            key_base = f"ip:{client_ip}"
        
        # Add endpoint path for endpoint-specific limiting
        path_hash = hashlib.md5(request.url.path.encode()).hexdigest()[:8]
        return f"rate_limit:{key_base}:{path_hash}"

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

    def _get_limits_for_path(self, path: str) -> Dict[str, int]:
        """
        Get rate limits for specific path.
        
        Returns:
            Dictionary with per_minute and burst limits
        """
        # Check for exact match first
        per_minute_limit = RateLimitConfig.ENDPOINT_LIMITS.get(path)
        
        # Check for wildcard matches
        if not per_minute_limit:
            for pattern, limit in RateLimitConfig.ENDPOINT_LIMITS.items():
                if pattern.endswith("/*") and path.startswith(pattern[:-2]):
                    per_minute_limit = limit
                    break
        
        # Use default if no specific limit found
        if not per_minute_limit:
            per_minute_limit = RateLimitConfig.DEFAULT_LIMIT
        
        # Get burst limits
        burst_limit, burst_window = RateLimitConfig.BURST_LIMITS.get(path, (per_minute_limit // 2, 60))
        
        return {
            "per_minute": per_minute_limit,
            "burst_limit": burst_limit,
            "burst_window": burst_window
        }

    async def _check_rate_limit(self, key: str, limits: Dict[str, int]) -> Tuple[bool, int]:
        """
        Check if request is within rate limits using sliding window.
        
        Args:
            key: Rate limit key
            limits: Rate limit configuration
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = datetime.utcnow()
        minute_key = f"{key}:minute:{now.minute}"
        burst_key = f"{key}:burst:{int(now.timestamp() // limits['burst_window'])}"
        
        try:
            if self.redis_client:
                return await self._check_rate_limit_redis(minute_key, burst_key, limits, now)
            else:
                return await self._check_rate_limit_memory(minute_key, burst_key, limits, now)
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, 0  # Allow request if check fails

    async def _check_rate_limit_redis(
        self, 
        minute_key: str, 
        burst_key: str, 
        limits: Dict[str, int], 
        now: datetime
    ) -> Tuple[bool, int]:
        """Check rate limits using Redis."""
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Check and increment minute counter
        pipe.incr(minute_key)
        pipe.expire(minute_key, 70)  # Expire after 70 seconds (buffer)
        
        # Check and increment burst counter
        pipe.incr(burst_key)
        pipe.expire(burst_key, limits["burst_window"] + 10)
        
        results = await pipe.execute()
        minute_count = results[0]
        burst_count = results[2]
        
        # Check limits
        if minute_count > limits["per_minute"]:
            return False, 60 - now.second  # Retry after remaining seconds in minute
        
        if burst_count > limits["burst_limit"]:
            return False, limits["burst_window"]  # Retry after burst window
        
        return True, 0

    async def _check_rate_limit_memory(
        self, 
        minute_key: str, 
        burst_key: str, 
        limits: Dict[str, int], 
        now: datetime
    ) -> Tuple[bool, int]:
        """Fallback rate limiting using in-memory store."""
        
        # Clean up expired entries
        self._cleanup_memory_store(now)
        
        # Get current counts
        minute_data = self.fallback_store.get(minute_key, {"count": 0, "expires": now + timedelta(minutes=1)})
        burst_data = self.fallback_store.get(burst_key, {"count": 0, "expires": now + timedelta(seconds=limits["burst_window"])})
        
        # Check if entries are expired
        if now > minute_data["expires"]:
            minute_data = {"count": 0, "expires": now + timedelta(minutes=1)}
        
        if now > burst_data["expires"]:
            burst_data = {"count": 0, "expires": now + timedelta(seconds=limits["burst_window"])}
        
        # Check limits
        if minute_data["count"] >= limits["per_minute"]:
            return False, int((minute_data["expires"] - now).total_seconds())
        
        if burst_data["count"] >= limits["burst_limit"]:
            return False, int((burst_data["expires"] - now).total_seconds())
        
        # Increment counters
        minute_data["count"] += 1
        burst_data["count"] += 1
        
        # Store updated data
        self.fallback_store[minute_key] = minute_data
        self.fallback_store[burst_key] = burst_data
        
        return True, 0

    def _cleanup_memory_store(self, now: datetime):
        """Clean up expired entries from memory store."""
        expired_keys = [
            key for key, data in self.fallback_store.items()
            if now > data.get("expires", now)
        ]
        
        for key in expired_keys:
            del self.fallback_store[key]

    async def _get_remaining_requests(self, key: str, limits: Dict[str, int]) -> int:
        """Get remaining requests for current window."""
        now = datetime.utcnow()
        minute_key = f"{key}:minute:{now.minute}"
        
        try:
            if self.redis_client:
                current_count = await self.redis_client.get(minute_key)
                current_count = int(current_count) if current_count else 0
            else:
                minute_data = self.fallback_store.get(minute_key, {"count": 0})
                current_count = minute_data["count"]
            
            return max(0, limits["per_minute"] - current_count)
        
        except Exception as e:
            logger.error(f"Failed to get remaining requests: {e}")
            return limits["per_minute"]

    async def _log_rate_limit_violation(self, request: Request, key: str):
        """Log rate limit violation for monitoring and analysis."""
        try:
            from supabase import create_client
            
            supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_ANON_KEY")
            )
            
            # Extract user ID if available
            user_id = None
            auth_header = request.headers.get("authorization")
            if auth_header:
                try:
                    from ...core.auth.token import decode_token
                    token = auth_header.split(" ")[1]
                    payload = decode_token(token)
                    if payload:
                        user_id = payload.get("sub")
                except:
                    pass
            
            # Log security event
            supabase.table("security_events").insert({
                "user_id": user_id,
                "ip": self._get_client_ip(request),
                "event_type": "rate_limit_exceeded",
                "meta": {
                    "path": request.url.path,
                    "method": request.method,
                    "rate_limit_key": key,
                    "user_agent": request.headers.get("user-agent", "")
                },
                "severity": "medium"
            }).execute()
            
        except Exception as e:
            logger.error(f"Failed to log rate limit violation: {e}")
