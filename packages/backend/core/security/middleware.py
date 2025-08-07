"""
Trust middleware for Zero Trust security enforcement.
Evaluates every request for trust score and policy compliance.
"""

import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import json

from .trust_engine import trust_engine, TrustContext
from .policy_manager import policy_manager
from ..auth.token import decode_token

logger = logging.getLogger(__name__)

class TrustMiddleware:
    """
    Middleware that enforces Zero Trust security policies.
    Calculates trust scores and evaluates access policies for each request.
    """
    
    def __init__(self, app):
        self.app = app
        
        # Paths that bypass trust evaluation
        self.exempt_paths = {
            "/auth/signup",
            "/auth/login", 
            "/auth/refresh",
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        }

    async def __call__(self, request: Request, call_next):
        """
        Process request through trust evaluation pipeline.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response with trust headers or error response
        """
        # Skip trust evaluation for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        try:
            # Extract authentication information
            auth_data = await self._extract_auth_data(request)
            if not auth_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Build trust context
            trust_context = await self._build_trust_context(request, auth_data)
            
            # Calculate trust score
            trust_score = await trust_engine.calculate_trust_score(trust_context)
            
            # Evaluate access policy
            policy_result = await policy_manager.evaluate_access(
                resource_path=request.url.path,
                method=request.method,
                trust_score=trust_score,
                user_context={
                    "ip": self._get_client_ip(request),
                    "country": trust_context.geo_country,
                    "user_agent": request.headers.get("user-agent", "")
                }
            )
            
            # Check if access is allowed
            if not policy_result.allowed:
                await self._log_security_event(
                    request, 
                    auth_data.get("user_id"),
                    "access_denied",
                    {
                        "trust_score": trust_score,
                        "required_trust": policy_result.trust_required,
                        "reason": policy_result.reason
                    }
                )
                
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "Access denied",
                        "trust_score": trust_score,
                        "required_trust": policy_result.trust_required,
                        "reason": policy_result.reason
                    },
                    headers={"Retry-After": "60"}
                )
            
            # Store trust information in request state
            request.state.trust_score = trust_score
            request.state.trust_context = trust_context
            request.state.policy_result = policy_result
            
            # Process request
            response = await call_next(request)
            
            # Add trust headers to response
            if isinstance(response, Response):
                response.headers["X-Trust-Score"] = str(trust_score)
                response.headers["X-Trust-Level"] = self._get_trust_level(trust_score)
                response.headers["X-Policy-Matched"] = str(bool(policy_result.policy_matched))
            
            # Log successful access
            await self._log_security_event(
                request,
                auth_data.get("user_id"),
                "access_granted",
                {"trust_score": trust_score}
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Trust middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Security evaluation failed"}
            )

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from trust evaluation."""
        return path in self.exempt_paths or path.startswith("/static/")

    async def _extract_auth_data(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Extract authentication data from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Authentication data or None
        """
        # Get Authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        
        # Decode token
        payload = decode_token(token)
        if not payload:
            return None
        
        # Get device fingerprint
        fp_header = request.headers.get("x-device-fingerprint")
        if not fp_header:
            return None
        
        return {
            "user_id": payload.get("sub"),
            "token_fp": payload.get("fp"),
            "request_fp": hashlib.sha256(fp_header.encode()).hexdigest(),
            "token_type": payload.get("type", "access")
        }

    async def _build_trust_context(self, request: Request, auth_data: Dict[str, Any]) -> TrustContext:
        """
        Build trust evaluation context from request data.
        
        Args:
            request: FastAPI request object
            auth_data: Authentication data
            
        Returns:
            Trust context for evaluation
        """
        # Verify device fingerprint matches token
        if auth_data["token_fp"] != auth_data["request_fp"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Device fingerprint mismatch"
            )
        
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Get geographical information (simplified - in production use proper GeoIP)
        geo_country = self._get_country_from_ip(client_ip)
        
        # Build context
        context = TrustContext(
            user_id=auth_data["user_id"],
            fp_hash=auth_data["token_fp"],
            ip_address=client_ip,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            geo_country=geo_country,
            is_vpn=self._detect_vpn(client_ip)
        )
        
        return context

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request headers."""
        # Check for forwarded headers (when behind proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        return request.client.host if request.client else "unknown"

    def _get_country_from_ip(self, ip: str) -> Optional[str]:
        """
        Get country code from IP address.
        Simplified implementation - use proper GeoIP service in production.
        """
        # This is a placeholder - in production, use MaxMind GeoIP2 or similar
        if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("127."):
            return "US"  # Assume local development
        
        # For demo purposes, assign countries based on IP ranges
        # In production, use actual GeoIP database
        try:
            octets = ip.split(".")
            if len(octets) == 4:
                first_octet = int(octets[0])
                if first_octet < 100:
                    return "US"
                elif first_octet < 150:
                    return "DE" 
                elif first_octet < 200:
                    return "GB"
                else:
                    return "CA"
        except:
            pass
        
        return "US"  # Default fallback

    def _detect_vpn(self, ip: str) -> bool:
        """
        Detect if IP is from VPN/proxy.
        Simplified implementation - use proper VPN detection service in production.
        """
        # Placeholder implementation
        # In production, use services like IPQualityScore, MaxMind, etc.
        known_vpn_ranges = [
            "45.134.", "91.219.", "185.220."  # Example VPN IP ranges
        ]
        
        return any(ip.startswith(range_prefix) for range_prefix in known_vpn_ranges)

    def _get_trust_level(self, score: int) -> str:
        """Convert trust score to human-readable level."""
        if score >= 80:
            return "high"
        elif score >= 60:
            return "medium"
        elif score >= 40:
            return "low"
        else:
            return "very_low"

    async def _log_security_event(
        self, 
        request: Request, 
        user_id: Optional[str],
        event_type: str, 
        metadata: Dict[str, Any]
    ):
        """Log security event to database."""
        try:
            from supabase import create_client
            import os
            
            supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_ANON_KEY")
            )
            
            supabase.table("security_events").insert({
                "user_id": user_id,
                "ip": self._get_client_ip(request),
                "event_type": event_type,
                "meta": metadata,
                "severity": "high" if event_type == "access_denied" else "medium"
            }).execute()
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    