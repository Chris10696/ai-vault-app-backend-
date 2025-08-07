"""
Trust scoring engine implementing contextual risk assessment.
Calculates trust scores based on device reputation, geo-velocity, and behavioral patterns.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis
import os
import geoip2.database
import geoip2.errors

logger = logging.getLogger(__name__)

class TrustFactors(Enum):
    """Trust factors that influence the overall trust score."""
    DEVICE_REPUTATION = "device_reputation"
    GEO_VELOCITY = "geo_velocity"
    FAILED_ATTEMPTS = "failed_attempts"  
    TIME_SINCE_LAST_LOGIN = "time_since_last_login"
    VPN_DETECTION = "vpn_detection"
    DEVICE_AGE = "device_age"
    SESSION_ANOMALY = "session_anomaly"

@dataclass
class TrustContext:
    """Context information used for trust score calculation."""
    user_id: str
    fp_hash: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    previous_login: Optional[datetime] = None
    failed_attempts: int = 0
    is_vpn: bool = False
    device_age_days: int = 0
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None

class TrustEngine:
    """
    Trust scoring engine implementing multi-factor risk assessment.
    Uses Redis for caching and performance optimization.
    """
    
    BASE_TRUST_SCORE = 50
    CACHE_TTL = int(os.getenv("TRUST_SCORE_TTL", 900))  # 15 minutes
    
    def __init__(self):
        """Initialize trust engine with Redis connection."""
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client: Optional[redis.Redis] = None
        self._trust_weights = {
            TrustFactors.DEVICE_REPUTATION: 0.25,
            TrustFactors.GEO_VELOCITY: 0.20,
            TrustFactors.FAILED_ATTEMPTS: 0.20,
            TrustFactors.TIME_SINCE_LAST_LOGIN: 0.10,
            TrustFactors.VPN_DETECTION: 0.15,
            TrustFactors.DEVICE_AGE: 0.05,
            TrustFactors.SESSION_ANOMALY: 0.05
        }
        
    async def initialize(self):
        """Initialize Redis connection and other resources."""
        try:
            self.redis_client = redis.from_url(self.redis_url, encoding="utf8", decode_responses=True)
            await self.redis_client.ping()
            logger.info("Trust engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trust engine: {e}")
            raise

    async def calculate_trust_score(self, context: TrustContext) -> int:
        """
        Calculate comprehensive trust score based on multiple factors.
        
        Args:
            context: Trust evaluation context
            
        Returns:
            Trust score between 0-100
        """
        cache_key = f"trust:{context.fp_hash}:{context.user_id}"
        
        # Try to get cached score first
        try:
            if self.redis_client:
                cached_score = await self.redis_client.get(cache_key)
                if cached_score:
                    logger.debug(f"Using cached trust score: {cached_score}")
                    return int(cached_score)
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")
        
        # Calculate new trust score
        score = self.BASE_TRUST_SCORE
        
        # Apply trust factors
        score += await self._evaluate_device_reputation(context)
        score += await self._evaluate_geo_velocity(context)
        score += await self._evaluate_failed_attempts(context)
        score += await self._evaluate_time_factors(context)
        score += await self._evaluate_vpn_detection(context)
        score += await self._evaluate_device_age(context)
        score += await self._evaluate_session_anomaly(context)
        
        # Clamp score to valid range
        final_score = max(0, min(100, score))
        
        # Cache the result
        try:
            if self.redis_client:
                await self.redis_client.setex(cache_key, self.CACHE_TTL, final_score)
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")
        
        logger.info(f"Trust score calculated: {final_score} for user {context.user_id}")
        return final_score

    async def _evaluate_device_reputation(self, context: TrustContext) -> int:
        """Evaluate device reputation based on historical behavior."""
        reputation_key = f"device_reputation:{context.fp_hash}"
        
        try:
            if self.redis_client:
                reputation_data = await self.redis_client.get(reputation_key)
                if reputation_data:
                    data = json.loads(reputation_data)
                    successful_logins = data.get("successful_logins", 0)
                    failed_logins = data.get("failed_logins", 0)
                    
                    if successful_logins + failed_logins > 0:
                        success_rate = successful_logins / (successful_logins + failed_logins)
                        if success_rate > 0.9:
                            return 15  # High reputation
                        elif success_rate > 0.7:
                            return 5   # Medium reputation
                        else:
                            return -10  # Low reputation
        except Exception as e:
            logger.warning(f"Device reputation evaluation failed: {e}")
        
        return 0  # Neutral for new devices

    async def _evaluate_geo_velocity(self, context: TrustContext) -> int:
        """Evaluate geographical velocity for impossible travel detection."""
        if not context.geo_country:
            return 0
            
        last_location_key = f"last_location:{context.user_id}"
        
        try:
            if self.redis_client:
                last_location_data = await self.redis_client.get(last_location_key)
                if last_location_data:
                    data = json.loads(last_location_data)
                    last_country = data.get("country")
                    last_timestamp = datetime.fromisoformat(data.get("timestamp"))
                    
                    # Check for impossible travel (simplified)
                    time_diff = (context.timestamp - last_timestamp).total_seconds()
                    
                    if last_country != context.geo_country and time_diff < 3600:  # 1 hour
                        return -20  # Suspicious geo-velocity
                    elif last_country == context.geo_country:
                        return 10   # Consistent location
                
                # Update last location
                await self.redis_client.setex(
                    last_location_key, 
                    86400,  # 24 hours
                    json.dumps({
                        "country": context.geo_country,
                        "city": context.geo_city,
                        "timestamp": context.timestamp.isoformat()
                    })
                )
        except Exception as e:
            logger.warning(f"Geo-velocity evaluation failed: {e}")
        
        # Trusted countries get bonus
        trusted_countries = {"US", "DE", "CA", "GB", "AU"}
        if context.geo_country in trusted_countries:
            return 5
        
        return 0

    async def _evaluate_failed_attempts(self, context: TrustContext) -> int:
        """Evaluate impact of recent failed login attempts."""
        if context.failed_attempts == 0:
            return 5  # No recent failures
        elif context.failed_attempts <= 2:
            return -5  # Few failures
        elif context.failed_attempts <= 5:
            return -15  # Multiple failures
        else:
            return -30  # Many failures - high risk

    async def _evaluate_time_factors(self, context: TrustContext) -> int:
        """Evaluate time-based trust factors."""
        if not context.previous_login:
            return -5  # First time login
        
        time_since_last = (context.timestamp - context.previous_login).total_seconds()
        
        if time_since_last < 300:  # 5 minutes
            return 10  # Recent activity
        elif time_since_last < 3600:  # 1 hour
            return 5   # Regular activity
        elif time_since_last < 86400:  # 1 day
            return 0   # Normal activity
        elif time_since_last < 604800:  # 1 week
            return -5  # Infrequent activity
        else:
            return -10  # Long absence

    async def _evaluate_vpn_detection(self, context: TrustContext) -> int:
        """Evaluate VPN/proxy usage impact on trust."""
        if context.is_vpn:
            return -25  # VPN usage reduces trust
        return 0

    async def _evaluate_device_age(self, context: TrustContext) -> int:
        """Evaluate device age factor."""
        if context.device_age_days > 30:
            return 5   # Established device
        elif context.device_age_days > 7:
            return 2   # Recent device
        else:
            return -3  # New device

    async def _evaluate_session_anomaly(self, context: TrustContext) -> int:
        """Evaluate session behavior anomalies."""
        # Simplified anomaly detection
        # In production, this would use ML models
        
        # Check for unusual user agent
        if "bot" in context.user_agent.lower() or "crawler" in context.user_agent.lower():
            return -15
        
        # Check for session timing anomalies
        current_hour = context.timestamp.hour
        if 2 <= current_hour <= 5:  # Unusual hours
            return -5
        
        return 0

    async def update_device_reputation(self, fp_hash: str, success: bool):
        """Update device reputation based on login outcome."""
        reputation_key = f"device_reputation:{fp_hash}"
        
        try:
            if self.redis_client:
                reputation_data = await self.redis_client.get(reputation_key)
                
                if reputation_data:
                    data = json.loads(reputation_data)
                else:
                    data = {"successful_logins": 0, "failed_logins": 0}
                
                if success:
                    data["successful_logins"] += 1
                else:
                    data["failed_logins"] += 1
                
                await self.redis_client.setex(
                    reputation_key,
                    86400 * 30,  # 30 days
                    json.dumps(data)
                )
        except Exception as e:
            logger.error(f"Failed to update device reputation: {e}")

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

# Global trust engine instance
trust_engine = TrustEngine()