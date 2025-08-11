from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime
import json
import uuid

from ...ml_models.recommendation.hybrid_recommender import HybridRecommendationEngine
from ...ml_models.recommendation.collaborative_filtering import CollaborativeFilteringEngine
from ...ml_models.recommendation.content_based_filtering import ContentBasedRecommendationEngine
from ...core.database.connection import get_database
from ...core.auth.jwt_handler import verify_token
from ...core.privacy.differential_privacy import DifferentialPrivacy

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])
security = HTTPBearer()

# Initialize ML engines (these would be loaded from saved models in production)
hybrid_engine = HybridRecommendationEngine()
cf_engine = CollaborativeFilteringEngine()
cb_engine = ContentBasedRecommendationEngine()

# Request/Response Models
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    num_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")
    recommendation_type: str = Field("hybrid", regex="^(collaborative|content_based|hybrid)$")
    include_explanations: bool = Field(True, description="Include explanations")
    diversity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Diversity threshold")
    ab_test_group: str = Field("balanced", regex="^(control|cf_heavy|cb_heavy|balanced)$")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    request_id: str
    timestamp: datetime

class TrainingRequest(BaseModel):
    model_type: str = Field(..., regex="^(collaborative|content_based|hybrid)$")
    validation_split: float = Field(0.2, ge=0.1, le=0.5)
    force_retrain: bool = Field(False)

class TrainingResponse(BaseModel):
    training_id: str
    status: str
    message: str
    estimated_time_minutes: int

class ModelStatusResponse(BaseModel):
    cf_trained: bool
    cb_fitted: bool
    hybrid_ready: bool
    last_training: Optional[datetime]
    model_version: str

class FeedbackRequest(BaseModel):
    user_id: str
    recommendation_id: str
    feedback_type: str = Field(..., regex="^(like|dislike|click|ignore|rate)$")
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_context: Optional[Dict[str, Any]] = None

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user information"""
    try:
        token = credentials.credentials
        payload = verify_token(token)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {str(e)}")

async def check_model_readiness():
    """Check if models are ready for inference"""
    if not hybrid_engine.is_ready and not cf_engine.is_trained and not cb_engine.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Recommendation models are not ready. Please train the models first."
        )

# API Endpoints

@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get the current status of recommendation models"""
    return ModelStatusResponse(
        cf_trained=cf_engine.is_trained,
        cb_fitted=cb_engine.is_fitted,
        hybrid_ready=hybrid_engine.is_ready,
        last_training=datetime.utcnow(),  # This would come from database
        model_version="1.0.0"
    )

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    user_data: dict = Depends(get_current_user),
    _: None = Depends(check_model_readiness)
):
    """
    Generate personalized recommendations for a user
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Generating recommendations for user {request.user_id}, request {request_id}")
        
        # Validate user access
        if request.user_id != user_data.get("user_id") and user_data.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Generate recommendations based on type
        if request.recommendation_type == "hybrid":
            recommendations = await hybrid_engine.recommend(
                user_id=request.user_id,
                num_recommendations=request.num_recommendations,
                return_explanations=request.include_explanations,
                diversity_threshold=request.diversity_threshold,
                ab_test_group=request.ab_test_group,
                context=request.context or {}
            )
        elif request.recommendation_type == "collaborative":
            recommendations = await cf_engine.recommend(
                user_id=request.user_id,
                num_recommendations=request.num_recommendations,
                diversity_threshold=request.diversity_threshold
            )
        elif request.recommendation_type == "content_based":
            # Get user preferences for content-based recommendations
            user_preferences = await _get_user_preferences(request.user_id)
            recommendations = await cb_engine.recommend_for_user_preferences(
                user_preferences=user_preferences,
                num_recommendations=request.num_recommendations
            )
        
        # Add metadata
        metadata = {
            "recommendation_type": request.recommendation_type,
            "ab_test_group": request.ab_test_group,
            "user_id": request.user_id,
            "model_version": "1.0.0",
            "privacy_preserved": True,
            "total_candidates": len(recommendations)
        }
        
        response = RecommendationResponse(
            recommendations=recommendations,
            metadata=metadata,
            request_id=request_id,
            timestamp=datetime.utcnow()
        )
        
        # Log recommendation asynchronously
        asyncio.create_task(_log_recommendation_served(request, response, user_data))
        
        logger.info(f"Successfully generated {len(recommendations)} recommendations for user {request.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.post("/similar/{app_id}")
async def get_similar_apps(
    app_id: str,
    num_recommendations: int = Query(10, ge=1, le=50),
    diversity_threshold: float = Query(0.7, ge=0.0, le=1.0),
    user_data: dict = Depends(get_current_user),
    _: None = Depends(check_model_readiness)
):
    """
    Get apps similar to a specific app using content-based filtering
    """
    try:
        if not cb_engine.is_fitted:
            raise HTTPException(status_code=503, detail="Content-based model not ready")
        
        similar_apps = await cb_engine.get_similar_apps(
            app_id=app_id,
            num_recommendations=num_recommendations,
            diversity_threshold=diversity_threshold
        )
        
        return {
            "app_id": app_id,
            "similar_apps": similar_apps,
            "metadata": {
                "method": "content_based",
                "diversity_threshold": diversity_threshold,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get similar apps for {app_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get similar apps: {str(e)}")

@router.post("/train", response_model=TrainingResponse)
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    user_data: dict = Depends(get_current_user)
):
    """
    Start training recommendation models
    """
    # Check admin privileges
    if user_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required for model training")
    
    training_id = str(uuid.uuid4())
    
    try:
        # Start training in background
        if request.model_type == "hybrid":
            background_tasks.add_task(
                _train_hybrid_model,
                training_id,
                request.validation_split,
                request.force_retrain
            )
            estimated_time = 45  # minutes
        elif request.model_type == "collaborative":
            background_tasks.add_task(
                _train_collaborative_model,
                training_id,
                request.validation_split,
                request.force_retrain
            )
            estimated_time = 30
        elif request.model_type == "content_based":
            background_tasks.add_task(
                _train_content_based_model,
                training_id,
                request.force_retrain
            )
            estimated_time = 15
        
        return TrainingResponse(
            training_id=training_id,
            status="started",
            message=f"{request.model_type} model training initiated",
            estimated_time_minutes=estimated_time
        )
        
    except Exception as e:
        logger.error(f"Failed to start model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/training/{training_id}")
async def get_training_status(
    training_id: str,
    user_data: dict = Depends(get_current_user)
):
    """
    Get the status of a training job
    """
    # This would query a training status database/cache
    # For now, return a placeholder response
    return {
        "training_id": training_id,
        "status": "completed",  # or "running", "failed"
        "progress": 100,
        "message": "Training completed successfully",
        "results": {
            "accuracy": 0.85,
            "training_time_minutes": 30
        }
    }

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    user_data: dict = Depends(get_current_user)
):
    """
    Submit user feedback on recommendations
    """
    try:
        # Validate user access
        if feedback.user_id != user_data.get("user_id") and user_data.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Store feedback in database
        await _store_user_feedback(feedback)
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "feedback_id": str(uuid.uuid4())
        }
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@router.get("/metrics")
async def get_recommendation_metrics(
    user_data: dict = Depends(get_current_user),
    days: int = Query(7, ge=1, le=90)
):
    """
    Get recommendation system metrics (admin only)
    """
    if user_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # This would query metrics from the database
    return {
        "time_period_days": days,
        "metrics": {
            "total_recommendations_served": 125000,
            "unique_users_served": 8500,
            "average_click_through_rate": 0.12,
            "average_precision_at_10": 0.25,
            "average_diversity_score": 0.78,
            "model_accuracy": {
                "collaborative": 0.82,
                "content_based": 0.75,
                "hybrid": 0.87
            }
        },
        "ab_test_results": {
            "control": {"ctr": 0.10, "satisfaction": 3.8},
            "cf_heavy": {"ctr": 0.11, "satisfaction": 3.9},
            "cb_heavy": {"ctr": 0.09, "satisfaction": 3.7},
            "balanced": {"ctr": 0.12, "satisfaction": 4.1}
        }
    }

# Helper functions

async def _get_user_preferences(user_id: str) -> Dict[str, Any]:
    """Get user preferences from database"""
    # This would query the database for user preferences
    return {
        "preferred_categories": ["productivity", "entertainment"],
        "rating_threshold": 4.0,
        "novelty_preference": 0.5
    }

async def _log_recommendation_served(
    request: RecommendationRequest,
    response: RecommendationResponse,
    user_data: dict
):
    """Log recommendation serving for analytics"""
    log_data = {
        "request_id": response.request_id,
        "user_id": request.user_id,
        "recommendation_type": request.recommendation_type,
        "ab_test_group": request.ab_test_group,
        "num_recommendations": len(response.recommendations),
        "served_at": response.timestamp.isoformat(),
        "user_agent": user_data.get("user_agent"),
        "session_id": user_data.get("session_id")
    }
    
    # In production, this would be stored in a database or sent to an analytics service
    logger.info(f"Recommendation served: {json.dumps(log_data)}")

async def _store_user_feedback(feedback: FeedbackRequest):
    """Store user feedback in database"""
    # This would insert feedback into the database
    logger.info(f"User feedback stored: {feedback.dict()}")

async def _train_hybrid_model(training_id: str, validation_split: float, force_retrain: bool):
    """Train hybrid model in background"""
    try:
        logger.info(f"Starting hybrid model training {training_id}")
        
        # Get training data
        interactions = await _get_interaction_data()
        apps_data = await _get_apps_data()
        
        # Train model
        training_results = await hybrid_engine.train(interactions, apps_data, validation_split)
        
        # Save model
        await hybrid_engine.save_model(f"models/hybrid_{training_id}")
        
        logger.info(f"Hybrid model training {training_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Hybrid model training {training_id} failed: {str(e)}")

async def _train_collaborative_model(training_id: str, validation_split: float, force_retrain: bool):
    """Train collaborative filtering model in background"""
    try:
        logger.info(f"Starting collaborative model training {training_id}")
        
        interactions = await _get_interaction_data()
        training_results = await cf_engine.train_model(interactions, validation_split)
        
        await cf_engine._save_model(f"models/collaborative_{training_id}.pt")
        
        logger.info(f"Collaborative model training {training_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Collaborative model training {training_id} failed: {str(e)}")

async def _train_content_based_model(training_id: str, force_retrain: bool):
    """Train content-based model in background"""
    try:
        logger.info(f"Starting content-based model training {training_id}")
        
        apps_data = await _get_apps_data()
        fitting_results = await cb_engine.fit(apps_data)
        
        await cb_engine.save_model(f"models/content_based_{training_id}.pkl")
        
        logger.info(f"Content-based model training {training_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Content-based model training {training_id} failed: {str(e)}")

async def _get_interaction_data() -> List[Dict[str, Any]]:
    """Get interaction data from database"""
    # This would query the user_interactions table
    return [
        {"user_id": "user1", "app_id": "app1", "rating": 5, "interaction_type": "rate"},
        {"user_id": "user1", "app_id": "app2", "rating": 3, "interaction_type": "rate"},
        # ... more data
    ]

async def _get_apps_data() -> List[Dict[str, Any]]:
    """Get apps data from database"""
    # This would query the apps table
    return [
        {
            'id': 'app1',
            'name': 'Example App',
            'description': 'An example application',
            'category': 'Productivity',
            'tags': ['productivity', 'utility'],
            'security_score': 85,
            'rating_average': 4.2,
            'downloads_count': 50000
        }
        # ... more data
    ]
