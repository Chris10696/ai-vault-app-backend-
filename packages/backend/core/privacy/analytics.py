"""
Privacy budget analytics and monitoring system.
Provides insights into privacy usage patterns and budget optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from supabase import create_client, Client
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class UsageStats:
    """Privacy budget usage statistics."""
    total_queries: int
    total_epsilon_consumed: float
    average_epsilon_per_query: float
    most_common_query_type: str
    peak_usage_hour: int
    budget_utilization_rate: float

@dataclass
class BudgetRecommendation:
    """Budget optimization recommendation."""
    recommended_daily_limit: float
    recommended_total_budget: float
    confidence_score: float
    reasoning: str

class PrivacyAnalytics:
    """
    Analytics engine for privacy budget usage and optimization.
    Provides insights and recommendations for budget management.
    """
    
    def __init__(self):
        self.supabase: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client."""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
            logger.info("PrivacyAnalytics initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PrivacyAnalytics: {e}")
    
    async def get_user_usage_stats(
        self, 
        user_id: str, 
        days_back: int = 30
    ) -> UsageStats:
        """
        Get usage statistics for a user over the specified period.
        
        Args:
            user_id: User identifier
            days_back: Number of days to look back
            
        Returns:
            UsageStats object with usage metrics
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Get query data
            query_result = self.supabase.table("privacy_queries").select(
                "query_type, epsilon_used, requested_at"
            ).eq("user_id", user_id).gte(
                "requested_at", start_date.isoformat()
            ).eq("status", "completed").execute()
            
            if not query_result.data:
                return UsageStats(
                    total_queries=0,
                    total_epsilon_consumed=0.0,
                    average_epsilon_per_query=0.0,
                    most_common_query_type="none",
                    peak_usage_hour=12,  # Default
                    budget_utilization_rate=0.0
                )
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(query_result.data)
            df['requested_at'] = pd.to_datetime(df['requested_at'])
            df['hour'] = df['requested_at'].dt.hour
            
            # Calculate statistics
            total_queries = len(df)
            total_epsilon = df['epsilon_used'].sum()
            avg_epsilon = df['epsilon_used'].mean()
            most_common_type = df['query_type'].mode().iloc[0] if len(df) > 0 else "none"
            peak_hour = df['hour'].mode().iloc[0] if len(df) > 0 else 12
            
            # Get user budget info for utilization rate
            profile_result = self.supabase.table("user_privacy_profiles").select(
                "daily_epsilon_limit, total_epsilon_budget, epsilon_used_total"
            ).eq("user_id", user_id).execute()
            
            utilization_rate = 0.0
            if profile_result.data:
                profile = profile_result.data[0]
                utilization_rate = (profile['epsilon_used_total'] / profile['total_epsilon_budget']) * 100
            
            return UsageStats(
                total_queries=total_queries,
                total_epsilon_consumed=float(total_epsilon),
                average_epsilon_per_query=float(avg_epsilon),
                most_common_query_type=most_common_type,
                peak_usage_hour=int(peak_hour),
                budget_utilization_rate=float(utilization_rate)
            )
            
        except Exception as e:
            logger.error(f"Failed to get usage stats for user {user_id}: {e}")
            raise
    
    async def get_budget_recommendations(
        self, 
        user_id: str
    ) -> BudgetRecommendation:
        """
        Generate budget optimization recommendations for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            BudgetRecommendation with optimized limits
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            # Get user's historical usage patterns
            usage_stats = await self.get_user_usage_stats(user_id, days_back=90)
            
            # Get current budget settings
            profile_result = self.supabase.table("user_privacy_profiles").select("*").eq("user_id", user_id).execute()
            
            if not profile_result.data:
                # Return default recommendations for new users
                return BudgetRecommendation(
                    recommended_daily_limit=2.0,
                    recommended_total_budget=10.0,
                    confidence_score=0.5,
                    reasoning="Default settings for new user with no usage history"
                )
            
            current_profile = profile_result.data[0]
            
            # Analyze usage patterns to make recommendations
            if usage_stats.total_queries == 0:
                return BudgetRecommendation(
                    recommended_daily_limit=current_profile['daily_epsilon_limit'],
                    recommended_total_budget=current_profile['total_epsilon_budget'],
                    confidence_score=0.3,
                    reasoning="No recent usage data available for optimization"
                )
            
            # Calculate daily usage pattern
            daily_queries = await self._get_daily_query_pattern(user_id)
            avg_daily_epsilon = np.mean([day['epsilon_used'] for day in daily_queries])
            max_daily_epsilon = np.max([day['epsilon_used'] for day in daily_queries]) if daily_queries else 0
            
            # Recommend daily limit with 20% buffer above max observed usage
            recommended_daily = max(0.5, max_daily_epsilon * 1.2)
            
            # Recommend total budget based on projected monthly usage
            monthly_projection = avg_daily_epsilon * 30
            recommended_total = max(10.0, monthly_projection * 1.5)  # 1.5x safety factor
            
            # Calculate confidence based on data availability
            confidence = min(0.95, usage_stats.total_queries / 100)  # Higher confidence with more data
            
            # Generate reasoning
            reasoning = self._generate_recommendation_reasoning(
                usage_stats, 
                current_profile, 
                recommended_daily, 
                recommended_total
            )
            
            return BudgetRecommendation(
                recommended_daily_limit=round(recommended_daily, 2),
                recommended_total_budget=round(recommended_total, 2),
                confidence_score=round(confidence, 2),
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations for user {user_id}: {e}")
            raise
    
    async def _get_daily_query_pattern(self, user_id: str, days_back: int = 30) -> List[Dict]:
        """Get daily query patterns for a user."""
        if not self.supabase:
            return []
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Get daily aggregated data
            result = self.supabase.rpc(
                'get_daily_privacy_usage',
                {
                    'p_user_id': user_id,
                    'p_start_date': start_date.date().isoformat(),
                    'p_end_date': end_date.date().isoformat()
                }
            ).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get daily pattern for user {user_id}: {e}")
            return []
    
    def _generate_recommendation_reasoning(
        self, 
        stats: UsageStats, 
        current_profile: Dict, 
        rec_daily: float, 
        rec_total: float
    ) -> str:
        """Generate human-readable reasoning for recommendations."""
        reasons = []
        
        # Daily limit reasoning
        current_daily = current_profile['daily_epsilon_limit']
        if rec_daily > current_daily * 1.1:
            reasons.append(f"Increase daily limit to {rec_daily} (current usage patterns exceed current limit)")
        elif rec_daily < current_daily * 0.9:
            reasons.append(f"Reduce daily limit to {rec_daily} to optimize privacy protection")
        else:
            reasons.append("Current daily limit is appropriate for your usage patterns")
        
        # Total budget reasoning
        current_total = current_profile['total_epsilon_budget']
        if rec_total > current_total * 1.1:
            reasons.append(f"Increase total budget to {rec_total} to accommodate growth")
        elif rec_total < current_total * 0.9:
            reasons.append(f"Reduce total budget to {rec_total} for better privacy control")
        
        # Usage pattern insights
        if stats.budget_utilization_rate > 80:
            reasons.append("High budget utilization detected - consider increasing limits")
        elif stats.budget_utilization_rate < 20:
            reasons.append("Low budget utilization - current limits may be too high")
        
        return ". ".join(reasons)
    
    async def get_system_analytics(self) -> Dict:
        """
        Get system-wide privacy analytics.
        
        Returns:
            System analytics dictionary
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            # Get overall statistics
            total_users = self.supabase.table("user_privacy_profiles").select("count", count="exact").execute()
            
            total_queries = self.supabase.table("privacy_queries").select("count", count="exact").execute()
            
            # Get query type distribution
            query_types = self.supabase.table("privacy_queries").select(
                "query_type"
            ).eq("status", "completed").execute()
            
            type_distribution = {}
            if query_types.data:
                df = pd.DataFrame(query_types.data)
                type_distribution = df['query_type'].value_counts().to_dict()
            
            # Get epsilon consumption trends
            recent_usage = self.supabase.table("privacy_audit_logs").select(
                "epsilon_consumed, timestamp"
            ).gte(
                "timestamp", (datetime.utcnow() - timedelta(days=7)).isoformat()
            ).execute()
            
            total_epsilon_week = 0.0
            if recent_usage.data:
                total_epsilon_week = sum(float(log['epsilon_consumed']) for log in recent_usage.data)
            
            return {
                "total_users": total_users.count if total_users.count else 0,
                "total_queries": total_queries.count if total_queries.count else 0,
                "query_type_distribution": type_distribution,
                "epsilon_consumed_last_7_days": round(total_epsilon_week, 4),
                "average_epsilon_per_query": round(total_epsilon_week / max(1, len(recent_usage.data or [])), 4),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system analytics: {e}")
            raise

# Add SQL function for daily usage aggregation
DAILY_USAGE_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION get_daily_privacy_usage(
    p_user_id UUID,
    p_start_date DATE,
    p_end_date DATE
)
RETURNS TABLE (
    usage_date DATE,
    query_count INTEGER,
    epsilon_used DECIMAL(10,6)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        DATE(pq.requested_at) as usage_date,
        COUNT(*)::INTEGER as query_count,
        COALESCE(SUM(pq.epsilon_used), 0)::DECIMAL(10,6) as epsilon_used
    FROM privacy_queries pq
    WHERE pq.user_id = p_user_id
      AND DATE(pq.requested_at) BETWEEN p_start_date AND p_end_date
      AND pq.status = 'completed'
    GROUP BY DATE(pq.requested_at)
    ORDER BY usage_date;
END;
$$ LANGUAGE plpgsql;
"""

# Global analytics instance
privacy_analytics = PrivacyAnalytics()
