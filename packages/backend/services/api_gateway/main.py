"""
Main FastAPI application entry point for AI Vault Backend
Integrates all modules including homomorphic encryption capabilities
"""

from datetime import datetime
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError as e:
    raise ImportError(f"FastAPI dependencies not installed: {e}")

# Import application modules
from .he_endpoints import he_router
from ..execution.he_inference_engine import HEInferenceEngine
from ..execution.inference_pipeline import AdvancedInferencePipeline
from ...core.crypto.seal_manager import SealContextManager
from ...core.crypto.key_manager import AdvancedKeyManager
from ...core.crypto.performance_optimizer import PerformanceOptimizer
from ...core.security.zero_trust import SecurityContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global application state
app_state = {
    'he_engine': None,
    'pipeline': None,
    'performance_optimizer': None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Initializes and cleans up resources
    """
    logger.info("Starting AI Vault Backend with Homomorphic Encryption...")
    
    try:
        # Initialize security context
        security_context = SecurityContext.create_default()
        
        # Initialize SEAL manager
        seal_manager = SealContextManager(security_context)
        logger.info("SEAL Context Manager initialized")
        
        # Initialize key manager
        key_manager = AdvancedKeyManager(security_context)
        logger.info("Advanced Key Manager initialized")
        
        # Initialize HE inference engine
        he_engine = HEInferenceEngine(security_context, seal_manager, key_manager)
        app_state['he_engine'] = he_engine
        logger.info("HE Inference Engine initialized")
        
        # Initialize inference pipeline
        pipeline = AdvancedInferencePipeline(security_context, he_engine)
        app_state['pipeline'] = pipeline
        logger.info("Advanced Inference Pipeline initialized")
        
        # Initialize performance optimizer
        performance_optimizer = PerformanceOptimizer(seal_manager, security_context)
        app_state['performance_optimizer'] = performance_optimizer
        logger.info("Performance Optimizer initialized")
        
        logger.info("All homomorphic encryption components initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Shutting down AI Vault Backend...")
        
        if app_state['performance_optimizer']:
            app_state['performance_optimizer'].shutdown()
        
        if app_state['pipeline']:
            app_state['pipeline'].cleanup_old_executions(0)  # Clean all
        
        if app_state['he_engine']:
            app_state['he_engine'].cleanup_inactive_sessions(0)  # Clean all
        
        logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="AI Vault Backend",
    description="Secure AI App Store with Homomorphic Encryption",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.aiault.com"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://aiault.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(he_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Vault Backend with Homomorphic Encryption",
        "version": "1.0.0",
        "features": [
            "Homomorphic Encryption",
            "Zero Trust Security",
            "Differential Privacy",
            "Performance Optimization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if core components are operational
        he_engine = app_state.get('he_engine')
        pipeline = app_state.get('pipeline')
        optimizer = app_state.get('performance_optimizer')
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "he_engine": "operational" if he_engine else "not_initialized",
                "pipeline": "operational" if pipeline else "not_initialized",
                "optimizer": "operational" if optimizer else "not_initialized"
            }
        }
        
        if he_engine:
            he_stats = he_engine.get_inference_statistics()
            health_status["he_statistics"] = {
                "loaded_models": he_stats.get("loaded_models", 0),
                "active_sessions": he_stats.get("active_sessions", 0),
                "total_requests": he_stats.get("total_requests", 0)
            }
        
        if optimizer:
            perf_report = optimizer.get_performance_report()
            health_status["performance"] = {
                "total_operations": perf_report.get("summary", {}).get("total_operations", 0),
                "avg_execution_time": perf_report.get("summary", {}).get("avg_execution_time", 0)
            }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_count": os.cpu_count(),
                "memory_usage": psutil.virtual_memory()._asdict()
            }
        }
        
        # Add HE-specific metrics
        he_engine = app_state.get('he_engine')
        if he_engine:
            metrics["homomorphic_encryption"] = he_engine.get_inference_statistics()
        
        # Add performance metrics
        optimizer = app_state.get('performance_optimizer')
        if optimizer:
            metrics["performance"] = optimizer.get_performance_report()
        
        # Add pipeline metrics
        pipeline = app_state.get('pipeline')
        if pipeline:
            metrics["pipeline"] = pipeline.get_pipeline_statistics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics collection failed")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Add startup message
@app.on_event("startup")
async def startup_message():
    """Log startup message"""
    logger.info("🔐 AI Vault Backend with Homomorphic Encryption is starting...")
    logger.info("🚀 Ready to serve secure AI inference requests!")

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
