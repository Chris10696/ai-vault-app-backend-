from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio
from datetime import datetime

from .recommendation_api import router as recommendation_router
from ...core.database.connection import init_database
from ...ml_models.recommendation.hybrid_recommender import HybridRecommendationEngine
from ...core.auth.jwt_handler import JWTHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global ML engines (would be managed better in production)
ml_engines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    # Startup
    logger.info("Starting Neural Discovery Engine Service")
    
    try:
        # Initialize database
        await init_database()
        
        # Load pre-trained models if available
        await load_pretrained_models()
        
        logger.info("Neural Discovery Engine Service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Neural Discovery Engine Service")
    
    # Save models if needed
    await save_models_on_shutdown()
    
    logger.info("Neural Discovery Engine Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="AI Vault - Neural Discovery Engine",
    description="ML-powered recommendation system with privacy preservation",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://aivault.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.aivault.app"]
)

# Include routers
app.include_router(recommendation_router)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Neural Discovery Engine",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    # Check database connection
    from ...core.database.connection import DatabaseManager
    db_healthy = await DatabaseManager.check_connection()
    
    # Check model status
    models_status = {
        "hybrid_ready": ml_engines.get("hybrid", {}).get("is_ready", False),
        "cf_trained": ml_engines.get("cf", {}).get("is_trained", False),
        "cb_fitted": ml_engines.get("cb", {}).get("is_fitted", False)
    }
    
    overall_health = db_healthy and any(models_status.values())
    
    return {
        "status": "healthy" if overall_health else "unhealthy",
        "database": "connected" if db_healthy else "disconnected",
        "models": models_status,
        "timestamp": datetime.utcnow().isoformat()
    }

async def load_pretrained_models():
    """Load pre-trained models on startup"""
    try:
        # This would load models from disk if they exist
        logger.info("Loading pre-trained models...")
        
        # Initialize engines
        hybrid_engine = HybridRecommendationEngine()
        
        # Try to load existing models
        # hybrid_engine.load_model("models/latest/hybrid/")
        
        ml_engines["hybrid"] = hybrid_engine
        
        logger.info("Pre-trained models loaded successfully")
        
    except Exception as e:
        logger.warning(f"Could not load pre-trained models: {e}")
        logger.info("Models will need to be trained before use")

async def save_models_on_shutdown():
    """Save models on application shutdown"""
    try:
        if "hybrid" in ml_engines and ml_engines["hybrid"].is_ready:
            await ml_engines["hybrid"].save_model("models/latest/hybrid/")
            logger.info("Models saved successfully")
    except Exception as e:
        logger.error(f"Failed to save models: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
