"""
FastAPI routes for differential privacy operations.
Provides endpoints for private queries, budget management, and user preferences.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
import logging

from ..auth.token import decode_token
from .models import PrivacyQuery, QueryType, PrivacyLevel, UserPrivacyProfile
from .budget import budget_manager
from .mechanism import LaplaceNoiseMechanism

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/privacy", tags=["privacy"])
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Extract user ID from JWT token."""
    try:
        payload = decode_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user ID"
            )
        
        return user_id
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed"
        )

@router.post("/query/count")
async def private_count_query(
    dataset_id: str,
    filter_params: Optional[Dict[str, Any]] = None,
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
    custom_epsilon: Optional[float] = None,
    user_id: str = Depends(get_current_user)
):
    """
    Execute a private count query with differential privacy.
    
    Args:
        dataset_id: Dataset identifier
        filter_params: Optional filter parameters
        privacy_level: Desired privacy level
        custom_epsilon: Custom epsilon value (overrides privacy level)
        user_id: Current user ID (from token)
        
    Returns:
        Private count result with noise
    """
    try:
        # Create privacy query
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.COUNT,
            dataset_id=dataset_id,
            query_params=filter_params or {},
            privacy_level=privacy_level,
            custom_epsilon=custom_epsilon
        )
        
        # For demo purposes, simulate a true count result
        # In production, this would query the actual dataset
        true_count = await _simulate_count_query(dataset_id, filter_params)
        
        # Execute private query
        response = await budget_manager.execute_private_query(query, true_count)
        
        return response
        
    except Exception as e:
        logger.error(f"Private count query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.post("/query/sum")
async def private_sum_query(
    dataset_id: str,
    column: str,
    filter_params: Optional[Dict[str, Any]] = None,
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
    custom_epsilon: Optional[float] = None,
    user_id: str = Depends(get_current_user)
):
    """
    Execute a private sum query with differential privacy.
    
    Args:
        dataset_id: Dataset identifier
        column: Column to sum
        filter_params: Optional filter parameters
        privacy_level: Desired privacy level
        custom_epsilon: Custom epsilon value
        user_id: Current user ID (from token)
        
    Returns:
        Private sum result with noise
    """
    try:
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.SUM,
            dataset_id=dataset_id,
            query_params={"column": column, "filters": filter_params or {}},
            privacy_level=privacy_level,
            custom_epsilon=custom_epsilon
        )
        
        # Simulate true sum result
        true_sum = await _simulate_sum_query(dataset_id, column, filter_params)
        
        response = await budget_manager.execute_private_query(query, true_sum)
        
        return response
        
    except Exception as e:
        logger.error(f"Private sum query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.post("/query/mean")
async def private_mean_query(
    dataset_id: str,
    column: str,
    filter_params: Optional[Dict[str, Any]] = None,
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
    custom_epsilon: Optional[float] = None,
    user_id: str = Depends(get_current_user)
):
    """
    Execute a private mean query with differential privacy.
    
    Args:
        dataset_id: Dataset identifier
        column: Column to calculate mean
        filter_params: Optional filter parameters
        privacy_level: Desired privacy level
        custom_epsilon: Custom epsilon value
        user_id: Current user ID (from token)
        
    Returns:
        Private mean result with noise
    """
    try:
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.MEAN,
            dataset_id=dataset_id,
            query_params={"column": column, "filters": filter_params or {}},
            privacy_level=privacy_level,
            custom_epsilon=custom_epsilon
        )
        
        # Simulate true mean result
        true_mean = await _simulate_mean_query(dataset_id, column, filter_params)
        
        response = await budget_manager.execute_private_query(query, true_mean)
        
        return response
        
    except Exception as e:
        logger.error(f"Private mean query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.get("/budget/status")
async def get_budget_status(
    user_id: str = Depends(get_current_user)
):
    """
    Get current privacy budget status for the user.
    
    Args:
        user_id: Current user ID (from token)
        
    Returns:
        Budget status including daily and total limits
    """
    try:
        status_info = await budget_manager.get_user_budget_status(user_id)
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get budget status: {str(e)}"
        )

@router.put("/preferences")
async def update_privacy_preferences(
    preferences: Dict[str, Any],
    user_id: str = Depends(get_current_user)
):
    """
    Update user privacy preferences.
    
    Args:
        preferences: Privacy preference updates
        user_id: Current user ID (from token)
        
    Returns:
        Success status
    """
    try:
        success = await budget_manager.update_user_preferences(user_id, preferences)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update preferences"
            )
        
        return {"message": "Preferences updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preferences: {str(e)}"
        )

@router.get("/privacy-levels")
async def get_privacy_levels():
    """
    Get available privacy levels and their epsilon values.
    
    Returns:
        Privacy levels with descriptions
    """
    return {
        "privacy_levels": {
            "minimal": {
                "epsilon": 1.0,
                "description": "Minimal privacy protection, higher accuracy"
            },
            "standard": {
                "epsilon": 0.5,
                "description": "Balanced privacy and accuracy (recommended)"
            },
            "high": {
                "epsilon": 0.1,
                "description": "High privacy protection, lower accuracy"
            },
            "maximum": {
                "epsilon": 0.01,
                "description": "Maximum privacy protection, significant noise"
            }
        },
        "custom_epsilon_range": {
            "min": 0.001,
            "max": 10.0,
            "description": "Custom epsilon values within this range"
        }
    }

@router.post("/simulate-accuracy")
async def simulate_accuracy_loss(
    query_type: QueryType,
    privacy_level: PrivacyLevel,
    custom_epsilon: Optional[float] = None,
    confidence: float = 0.95
):
    """
    Simulate expected accuracy loss for a query configuration.
    
    Args:
        query_type: Type of query
        privacy_level: Privacy level
        custom_epsilon: Custom epsilon value
        confidence: Confidence level for accuracy bound
        
    Returns:
        Expected accuracy loss estimates
    """
    try:
        mechanism = LaplaceNoiseMechanism()
        
        privacy_params = mechanism.get_privacy_parameters(
            query_type,
            privacy_level,
            custom_epsilon
        )
        
        accuracy_loss = mechanism.estimate_accuracy_loss(privacy_params, confidence)
        
        return {
            "query_type": query_type,
            "epsilon": privacy_params.epsilon,
            "sensitivity": privacy_params.sensitivity,
            "expected_accuracy_loss": accuracy_loss,
            "confidence_level": confidence,
            "description": f"With {confidence*100}% confidence, noise magnitude will be ≤ {accuracy_loss:.4f}"
        }
        
    except Exception as e:
        logger.error(f"Accuracy simulation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )

# Helper functions for simulating query results
# In production, these would query real datasets

async def _simulate_count_query(dataset_id: str, filter_params: Optional[Dict]) -> int:
    """Simulate a count query result."""
    # Simple simulation based on dataset_id
    base_count = hash(dataset_id) % 10000 + 1000
    if filter_params:
        # Reduce count based on filters
        filter_reduction = len(str(filter_params)) % 500
        base_count = max(100, base_count - filter_reduction)
    
    return base_count

async def _simulate_sum_query(dataset_id: str, column: str, filter_params: Optional[Dict]) -> float:
    """Simulate a sum query result."""
    base_sum = (hash(f"{dataset_id}:{column}") % 100000) + 10000.0
    if filter_params:
        filter_reduction = (len(str(filter_params)) % 10000)
        base_sum = max(1000.0, base_sum - filter_reduction)
    
    return base_sum

async def _simulate_mean_query(dataset_id: str, column: str, filter_params: Optional[Dict]) -> float:
    """Simulate a mean query result."""
    base_mean = ((hash(f"{dataset_id}:{column}") % 1000) / 10.0) + 50.0
    if filter_params:
        filter_adjustment = (len(str(filter_params)) % 20) - 10
        base_mean += filter_adjustment
    
    return max(0.1, base_mean)

@router.get("/health")
async def privacy_health_check():
    """Health check endpoint for privacy service."""
    return {
        "status": "healthy",
        "service": "privacy",
        "components": {
            "budget_manager": "active",
            "laplace_mechanism": "active",
            "redis_cache": "active"
        }
    }
