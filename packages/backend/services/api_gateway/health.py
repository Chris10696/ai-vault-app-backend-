"""
Health check and monitoring endpoints for API Gateway.
Provides detailed health status and metrics for monitoring systems.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import redis.asyncio as redis
from supabase import create_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check including all dependencies.
    Used by monitoring systems for comprehensive status.
    """
    health_status = {
        "service": "api_gateway",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {}
    }
    
    # Check Redis connection
    try:
        redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        await redis_client.ping()
        health_status["dependencies"]["redis"] = {"status": "healthy"}
        await redis_client.close()
    except Exception as e:
        health_status["dependencies"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check Supabase connection
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )
        # Simple query to test connection
        result = supabase.table("users").select("count", count="exact").limit(1).execute()
        health_status["dependencies"]["supabase"] = {"status": "healthy"}
    except Exception as e:
        health_status["dependencies"]["supabase"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] in ["healthy", "degraded"] else 503
    return JSONResponse(content=health_status, status_code=status_code)

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    return {"status": "ready"}

@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}
