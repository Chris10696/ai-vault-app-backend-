"""
Privacy budget management system for tracking and enforcing privacy loss.
Implements per-user budget tracking with time-based resets and composition rules.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import redis.asyncio as redis
from supabase import create_client, Client
import os

from .models import UserPrivacyProfile, PrivacyQuery, PrivacyAuditLog, PrivacyLevel
from .mechanism import LaplaceNoiseMechanism

logger = logging.getLogger(__name__)

class BudgetTracker:
    """
    Tracks privacy budget consumption for individual users.
    Maintains real-time budget information with persistent storage.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.supabase: Optional[Client] = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Redis and Supabase clients."""
        try:
            # Initialize Redis for fast budget tracking
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Initialize Supabase for persistent storage
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
            
            logger.info("BudgetTracker clients initialized")
        except Exception as e:
            logger.error(f"Failed to initialize BudgetTracker clients: {e}")
    
    async def get_user_profile(self, user_id: str) -> UserPrivacyProfile:
        """
        Get user privacy profile, creating default if not exists.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPrivacyProfile object
        """
        try:
            # Try Redis cache first
            if self.redis_client:
                cached_profile = await self.redis_client.get(f"privacy_profile:{user_id}")
                if cached_profile:
                    profile_data = json.loads(cached_profile)
                    profile = UserPrivacyProfile(**profile_data)
                    
                    # Check if daily reset is needed
                    if self._needs_daily_reset(profile):
                        profile = await self._reset_daily_budget(profile)
                    
                    return profile
            
            # Fallback to database
            if self.supabase:
                result = self.supabase.table("user_privacy_profiles").select("*").eq("user_id", user_id).execute()
                
                if result.data:
                    profile_data = result.data[0]
                    profile = UserPrivacyProfile(**profile_data)
                    
                    # Check if daily reset is needed
                    if self._needs_daily_reset(profile):
                        profile = await self._reset_daily_budget(profile)
                    
                    # Cache the profile
                    await self._cache_profile(profile)
                    return profile
            
            # Create default profile if not found
            profile = UserPrivacyProfile(user_id=user_id)
            await self._save_profile(profile)
            await self._cache_profile(profile)
            
            logger.info(f"Created default privacy profile for user {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get user profile for {user_id}: {e}")
            # Return default profile as fallback
            return UserPrivacyProfile(user_id=user_id)
    
    def _needs_daily_reset(self, profile: UserPrivacyProfile) -> bool:
        """Check if daily budget reset is needed."""
        now = datetime.utcnow()
        last_reset = profile.last_reset_date
        
        # Reset if it's a new day
        return now.date() > last_reset.date()
    
    async def _reset_daily_budget(self, profile: UserPrivacyProfile) -> UserPrivacyProfile:
        """Reset daily privacy budget."""
        profile.epsilon_used_today = 0.0
        profile.last_reset_date = datetime.utcnow()
        profile.updated_at = datetime.utcnow()
        
        await self._save_profile(profile)
        await self._cache_profile(profile)
        
        logger.info(f"Reset daily budget for user {profile.user_id}")
        return profile
    
    async def _save_profile(self, profile: UserPrivacyProfile):
        """Save profile to database."""
        if not self.supabase:
            return
        
        try:
            profile_dict = profile.dict()
            # Convert datetime objects to ISO strings
            for key, value in profile_dict.items():
                if isinstance(value, datetime):
                    profile_dict[key] = value.isoformat()
            
            # Check if profile exists
            existing = self.supabase.table("user_privacy_profiles").select("user_id").eq("user_id", profile.user_id).execute()
            
            if existing.data:
                # Update existing profile
                self.supabase.table("user_privacy_profiles").update(profile_dict).eq("user_id", profile.user_id).execute()
            else:
                # Insert new profile
                self.supabase.table("user_privacy_profiles").insert(profile_dict).execute()
                
        except Exception as e:
            logger.error(f"Failed to save profile for user {profile.user_id}: {e}")
    
    async def _cache_profile(self, profile: UserPrivacyProfile):
        """Cache profile in Redis."""
        if not self.redis_client:
            return
        
        try:
            profile_dict = profile.dict()
            # Convert datetime objects to ISO strings for JSON serialization
            for key, value in profile_dict.items():
                if isinstance(value, datetime):
                    profile_dict[key] = value.isoformat()
            
            await self.redis_client.setex(
                f"privacy_profile:{profile.user_id}",
                3600,  # 1 hour TTL
                json.dumps(profile_dict)
            )
        except Exception as e:
            logger.error(f"Failed to cache profile for user {profile.user_id}: {e}")
    
    async def check_budget_availability(
        self, 
        user_id: str, 
        requested_epsilon: float
    ) -> Tuple[bool, str, float]:
        """
        Check if user has sufficient privacy budget.
        
        Args:
            user_id: User identifier
            requested_epsilon: Requested epsilon value
            
        Returns:
            Tuple of (is_available, reason, remaining_budget)
        """
        profile = await self.get_user_profile(user_id)
        
        # Check daily limit
        if profile.epsilon_used_today + requested_epsilon > profile.daily_epsilon_limit:
            remaining_daily = profile.daily_epsilon_limit - profile.epsilon_used_today
            return False, f"Daily budget exceeded. Remaining: {remaining_daily:.4f}", remaining_daily
        
        # Check total limit
        if profile.epsilon_used_total + requested_epsilon > profile.total_epsilon_budget:
            remaining_total = profile.total_epsilon_budget - profile.epsilon_used_total
            return False, f"Total budget exceeded. Remaining: {remaining_total:.4f}", remaining_total
        
        # Budget is available
        remaining = min(
            profile.daily_epsilon_limit - profile.epsilon_used_today,
            profile.total_epsilon_budget - profile.epsilon_used_total
        )
        
        return True, "Budget available", remaining
    
    async def consume_budget(
        self, 
        user_id: str, 
        epsilon_used: float, 
        query_id: Optional[str] = None
    ) -> UserPrivacyProfile:
        """
        Consume privacy budget for a user.
        
        Args:
            user_id: User identifier
            epsilon_used: Epsilon value consumed
            query_id: Associated query ID for audit trail
            
        Returns:
            Updated UserPrivacyProfile
        """
        profile = await self.get_user_profile(user_id)
        
        # Update budget consumption
        profile.epsilon_used_today += epsilon_used
        profile.epsilon_used_total += epsilon_used
        profile.updated_at = datetime.utcnow()
        
        # Save updated profile
        await self._save_profile(profile)
        await self._cache_profile(profile)
        
        # Create audit log entry
        await self._create_audit_log(user_id, "budget_consumed", epsilon_used, query_id)
        
        logger.info(f"Consumed {epsilon_used:.4f} epsilon for user {user_id}")
        return profile
    
    async def _create_audit_log(
        self, 
        user_id: str, 
        operation_type: str, 
        epsilon_consumed: float,
        query_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Create audit log entry."""
        if not self.supabase:
            return
        
        try:
            audit_entry = PrivacyAuditLog(
                user_id=user_id,
                operation_type=operation_type,
                query_id=query_id,
                epsilon_consumed=epsilon_consumed,
                metadata=metadata or {}
            )
            
            audit_dict = audit_entry.dict()
            # Convert datetime to ISO string
            audit_dict['timestamp'] = audit_entry.timestamp.isoformat()
            
            self.supabase.table("privacy_audit_logs").insert(audit_dict).execute()
            
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")

class PrivacyBudgetManager:
    """
    High-level privacy budget management with policy enforcement.
    Coordinates budget tracking, policy evaluation, and privacy guarantees.
    """
    
    def __init__(self):
        self.budget_tracker = BudgetTracker()
        self.mechanism = LaplaceNoiseMechanism(secure_random=True)
        
        # Budget policies
        self.DEFAULT_DAILY_LIMIT = 2.0
        self.DEFAULT_TOTAL_BUDGET = 10.0
        self.MIN_EPSILON = 0.001
        self.MAX_EPSILON = 10.0
        
        logger.info("PrivacyBudgetManager initialized")
    
    async def request_privacy_budget(
        self, 
        query: PrivacyQuery
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Request privacy budget for a query.
        
        Args:
            query: Privacy query request
            
        Returns:
            Tuple of (approved, reason, epsilon_allocated)
        """
        try:
            # Get privacy parameters
            privacy_params = self.mechanism.get_privacy_parameters(
                query.query_type,
                query.privacy_level,
                query.custom_epsilon
            )
            
            requested_epsilon = privacy_params.epsilon
            
            # Validate epsilon bounds
            if requested_epsilon < self.MIN_EPSILON or requested_epsilon > self.MAX_EPSILON:
                return False, f"Epsilon must be between {self.MIN_EPSILON} and {self.MAX_EPSILON}", None
            
            # Check budget availability
            available, reason, remaining = await self.budget_tracker.check_budget_availability(
                query.user_id,
                requested_epsilon
            )
            
            if not available:
                return False, reason, None
            
            return True, "Budget approved", requested_epsilon
            
        except Exception as e:
            logger.error(f"Failed to request privacy budget: {e}")
            return False, f"Budget request failed: {str(e)}", None
    
    async def execute_private_query(self, query: PrivacyQuery, true_result: float) -> Dict:
        """
        Execute a private query with budget management.
        
        Args:
            query: Privacy query
            true_result: True query result
            
        Returns:
            Privacy response dictionary
        """
        try:
            # Request budget
            approved, reason, epsilon_allocated = await self.request_privacy_budget(query)
            
            if not approved:
                return {
                    "success": False,
                    "error": reason,
                    "query_id": query.query_id
                }
            
            # Get privacy parameters
            privacy_params = self.mechanism.get_privacy_parameters(
                query.query_type,
                query.privacy_level,
                epsilon_allocated
            )
            
            # Add noise to result
            noisy_result, noise_magnitude = self.mechanism.add_noise(true_result, privacy_params)
            
            # Consume budget
            updated_profile = await self.budget_tracker.consume_budget(
                query.user_id,
                epsilon_allocated,
                query.query_id
            )
            
            # Calculate accuracy loss
            accuracy_loss = self.mechanism.estimate_accuracy_loss(privacy_params)
            
            # Create response
            response = {
                "success": True,
                "query_id": query.query_id,
                "result": noisy_result,
                "epsilon_used": epsilon_allocated,
                "noise_magnitude": noise_magnitude,
                "accuracy_loss": accuracy_loss,
                "remaining_daily_budget": updated_profile.daily_epsilon_limit - updated_profile.epsilon_used_today,
                "remaining_total_budget": updated_profile.total_epsilon_budget - updated_profile.epsilon_used_total,
                "responded_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Executed private query {query.query_id} for user {query.user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to execute private query {query.query_id}: {e}")
            return {
                "success": False,
                "error": f"Query execution failed: {str(e)}",
                "query_id": query.query_id
            }
    
    async def get_user_budget_status(self, user_id: str) -> Dict:
        """
        Get current budget status for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Budget status dictionary
        """
        try:
            profile = await self.budget_tracker.get_user_profile(user_id)
            
            return {
                "user_id": user_id,
                "daily_budget": {
                    "limit": profile.daily_epsilon_limit,
                    "used": profile.epsilon_used_today,
                    "remaining": profile.daily_epsilon_limit - profile.epsilon_used_today,
                    "utilization": (profile.epsilon_used_today / profile.daily_epsilon_limit) * 100
                },
                "total_budget": {
                    "limit": profile.total_epsilon_budget,
                    "used": profile.epsilon_used_total,
                    "remaining": profile.total_epsilon_budget - profile.epsilon_used_total,
                    "utilization": (profile.epsilon_used_total / profile.total_epsilon_budget) * 100
                },
                "last_reset": profile.last_reset_date.isoformat(),
                "preferred_privacy_level": profile.preferred_privacy_level
            }
            
        except Exception as e:
            logger.error(f"Failed to get budget status for user {user_id}: {e}")
            return {"error": f"Failed to get budget status: {str(e)}"}
    
    async def update_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict
    ) -> bool:
        """
        Update user privacy preferences.
        
        Args:
            user_id: User identifier
            preferences: Preference updates
            
        Returns:
            True if successful
        """
        try:
            profile = await self.budget_tracker.get_user_profile(user_id)
            
            # Update allowed preferences
            if "preferred_privacy_level" in preferences:
                profile.preferred_privacy_level = PrivacyLevel(preferences["preferred_privacy_level"])
            
            if "daily_epsilon_limit" in preferences:
                new_limit = float(preferences["daily_epsilon_limit"])
                if 0 < new_limit <= profile.total_epsilon_budget:
                    profile.daily_epsilon_limit = new_limit
            
            profile.updated_at = datetime.utcnow()
            
            # Save updated profile
            await self.budget_tracker._save_profile(profile)
            await self.budget_tracker._cache_profile(profile)
            
            logger.info(f"Updated preferences for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update preferences for user {user_id}: {e}")
            return False

# Global budget manager instance
budget_manager = PrivacyBudgetManager()
