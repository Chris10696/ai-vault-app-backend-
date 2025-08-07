"""
Policy management system for Zero Trust access control.
Manages trust policies and evaluates access requests against defined rules.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from supabase import create_client, Client
import os
import json

logger = logging.getLogger(__name__)

class PolicyEffect(Enum):
    """Policy evaluation effects."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_MFA = "require_mfa"

@dataclass
class TrustPolicy:
    """Trust policy definition."""
    id: str
    resource_pattern: str
    required_trust_level: int
    conditions: Dict[str, Any]
    effect: PolicyEffect = PolicyEffect.ALLOW
    description: Optional[str] = None

@dataclass
class PolicyEvaluationResult:
    """Result of policy evaluation."""
    allowed: bool
    trust_required: int
    trust_actual: int
    policy_matched: Optional[TrustPolicy] = None
    reason: str = ""

class PolicyManager:
    """
    Manages Zero Trust policies and evaluates access requests.
    Integrates with Supabase for policy storage and management.
    """
    
    def __init__(self):
        """Initialize policy manager with Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase configuration missing")
            
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self._policy_cache: Dict[str, List[TrustPolicy]] = {}
        self._cache_ttl = 300  # 5 minutes
        logger.info("PolicyManager initialized")

    async def load_policies(self) -> List[TrustPolicy]:
        """
        Load all trust policies from database.
        
        Returns:
            List of trust policies
        """
        try:
            result = self.supabase.table("zero_trust_policies").select("*").execute()
            
            policies = []
            for row in result.data:
                policy = TrustPolicy(
                    id=row["id"],
                    resource_pattern=row["resource_pattern"],
                    required_trust_level=row["required_trust_level"],
                    conditions=row.get("conditions", {}),
                    description=row.get("description", "")
                )
                policies.append(policy)
            
            logger.info(f"Loaded {len(policies)} trust policies")
            return policies
            
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            return []

    async def find_matching_policy(self, resource_path: str, method: str) -> Optional[TrustPolicy]:
        """
        Find policy that matches the resource path and method.
        
        Args:
            resource_path: API resource path
            method: HTTP method
            
        Returns:
            Matching policy or None
        """
        policies = await self.load_policies()
        
        for policy in policies:
            if self._matches_pattern(resource_path, policy.resource_pattern):
                # Check method conditions
                allowed_methods = policy.conditions.get("methods", [])
                if not allowed_methods or method.upper() in allowed_methods:
                    logger.debug(f"Policy matched: {policy.resource_pattern} for {resource_path}")
                    return policy
        
        return None

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """
        Check if path matches pattern (supports wildcards).
        
        Args:
            path: Resource path to check
            pattern: Pattern with wildcards (*)
            
        Returns:
            True if path matches pattern
        """
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

    async def evaluate_access(
        self, 
        resource_path: str, 
        method: str, 
        trust_score: int,
        user_context: Dict[str, Any] = None
    ) -> PolicyEvaluationResult:
        """
        Evaluate access request against trust policies.
        
        Args:
            resource_path: API resource path
            method: HTTP method
            trust_score: Current user trust score
            user_context: Additional user context
            
        Returns:
            Policy evaluation result
        """
        policy = await self.find_matching_policy(resource_path, method)
        
        if not policy:
            # Default policy: allow with minimum trust
            return PolicyEvaluationResult(
                allowed=trust_score >= 30,  # Default minimum trust
                trust_required=30,
                trust_actual=trust_score,
                reason="Default policy applied - no specific policy found"
            )
        
        # Evaluate additional conditions
        if not self._evaluate_conditions(policy.conditions, user_context or {}):
            return PolicyEvaluationResult(
                allowed=False,
                trust_required=policy.required_trust_level,
                trust_actual=trust_score,
                policy_matched=policy,
                reason="Policy conditions not met"
            )
        
        # Check trust level requirement
        allowed = trust_score >= policy.required_trust_level
        
        return PolicyEvaluationResult(
            allowed=allowed,
            trust_required=policy.required_trust_level,
            trust_actual=trust_score,
            policy_matched=policy,
            reason=f"Trust score {'sufficient' if allowed else 'insufficient'} for policy"
        )

    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluate additional policy conditions.
        
        Args:
            conditions: Policy conditions
            context: User context
            
        Returns:
            True if conditions are met
        """
        # IP restrictions
        if "ips_not_allowed" in conditions:
            user_ip = context.get("ip")
            if user_ip:
                for blocked_ip_pattern in conditions["ips_not_allowed"]:
                    if self._matches_pattern(user_ip, blocked_ip_pattern):
                        return False
        
        # Time restrictions
        if "allowed_hours" in conditions:
            from datetime import datetime
            current_hour = datetime.utcnow().hour
            allowed_hours = conditions["allowed_hours"]
            if current_hour not in allowed_hours:
                return False
        
        # Country restrictions
        if "allowed_countries" in conditions:
            user_country = context.get("country")
            if user_country and user_country not in conditions["allowed_countries"]:
                return False
        
        return True

    async def create_policy(
        self, 
        resource_pattern: str, 
        required_trust_level: int,
        conditions: Dict[str, Any] = None,
        description: str = ""
    ) -> str:
        """
        Create a new trust policy.
        
        Args:
            resource_pattern: Resource pattern (supports wildcards)
            required_trust_level: Minimum trust level required
            conditions: Additional conditions
            description: Policy description
            
        Returns:
            Policy ID
        """
        try:
            result = self.supabase.table("zero_trust_policies").insert({
                "resource_pattern": resource_pattern,
                "required_trust_level": required_trust_level,
                "conditions": conditions or {},
                "description": description
            }).execute()
            
            if result.data:
                policy_id = result.data[0]["id"]
                logger.info(f"Created trust policy: {policy_id}")
                return policy_id
            else:
                raise Exception("Failed to create policy")
                
        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            raise

    async def update_policy(
        self, 
        policy_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update existing trust policy.
        
        Args:
            policy_id: Policy identifier
            updates: Fields to update
            
        Returns:
            True if successful
        """
        try:
            result = self.supabase.table("zero_trust_policies").update(updates).eq("id", policy_id).execute()
            
            if result.data:
                logger.info(f"Updated trust policy: {policy_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update policy {policy_id}: {e}")
            return False

    async def delete_policy(self, policy_id: str) -> bool:
        """
        Delete trust policy.
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            True if successful
        """
        try:
            result = self.supabase.table("zero_trust_policies").delete().eq("id", policy_id).execute()
            
            if result.data:
                logger.info(f"Deleted trust policy: {policy_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete policy {policy_id}: {e}")
            return False

# Global policy manager instance
policy_manager = PolicyManager()
