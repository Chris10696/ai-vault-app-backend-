"""
Main API Gateway application for AI Vault.
Updated to include Module B: Differential Privacy Engine integration.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
import uvicorn

# Import core modules from Module A
from ...core.auth.routes import router as auth_router
from ...core.security.middleware import TrustMiddleware
from ...core.security.trust_engine import trust_engine
from .rate_limiter import RateLimitMiddleware
from .health import router as health_router

# Import Module B privacy components
from ...core.privacy.routes import router as privacy_router
from ...core.privacy.vault_routes import router as vault_router
from ...core.privacy.monitoring import budget_monitor
from .privacy_integration import integrate_privacy_modules, setup_privacy_endpoints

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks for Redis connections and other resources.
    """
    # Startup
    try:
        # Initialize Redis connection for rate limiting
        redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf8",
            decode_responses=True
        )
        
        # Initialize FastAPI Limiter
        await FastAPILimiter.init(redis_client, prefix="rate_limit")
        
        # Initialize trust engine (Module A)
        await trust_engine.initialize()
        
        # Initialize privacy modules (Module B)
        integrate_privacy_modules(app)
        
        logger.info("API Gateway started successfully with privacy modules")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API Gateway: {e}")
        raise
    
    # Shutdown
    try:
        await FastAPILimiter.close()
        await trust_engine.close()
        budget_monitor.stop_monitoring()
        logger.info("API Gateway shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="AI Vault API Gateway",
    description="Secure API Gateway for AI Vault with Zero Trust architecture and Differential Privacy",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Add trust middleware (Zero Trust security)
app.add_middleware(TrustMiddleware)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "service": "api_gateway",
        "version": "1.0.0",
        "modules": {
            "authentication": "active",
            "zero_trust": "active", 
            "differential_privacy": "active",
            "encrypted_vault": "active"
        }
    }

# Include routers
app.include_router(auth_router, prefix="/api")
app.include_router(privacy_router, prefix="/api")
app.include_router(vault_router, prefix="/api") 
app.include_router(health_router, prefix="/api")

# Setup additional privacy endpoints
setup_privacy_endpoints(app)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Vault API Gateway with Privacy Engine",
        "version": "1.0.0", 
        "docs": "/docs",
        "health": "/health",
        "modules": {
            "auth": "/api/auth",
            "privacy": "/api/privacy",
            "vault": "/api/vault"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
