"""
Privacy module integration with API Gateway.
Adds privacy routes and middleware to the main API gateway.
"""

from fastapi import FastAPI
import logging

from ...core.privacy.routes import router as privacy_router
from ...core.privacy.vault_routes import router as vault_router
from ...core.privacy.monitoring import budget_monitor

logger = logging.getLogger(__name__)

def integrate_privacy_modules(app: FastAPI):
    """
    Integrate privacy modules with the API Gateway.
    
    Args:
        app: FastAPI application instance
    """
    try:
        # Add privacy routes
        app.include_router(privacy_router, prefix="/api")
        app.include_router(vault_router, prefix="/api")
        
        logger.info("Privacy modules integrated with API Gateway")
        
        # Start budget monitoring (optional - can be run separately)
        import asyncio
        async def start_monitoring():
            try:
                await budget_monitor.start_monitoring()
            except Exception as e:
                logger.error(f"Budget monitoring failed: {e}")
        
        # Note: In production, run this as a separate service
        # asyncio.create_task(start_monitoring())
        
    except Exception as e:
        logger.error(f"Failed to integrate privacy modules: {e}")
        raise

def setup_privacy_endpoints(app: FastAPI):
    """Setup additional privacy-related endpoints."""
    
    @app.get("/api/privacy/info")
    async def privacy_info():
        """Get privacy system information."""
        return {
            "differential_privacy": {
                "mechanism": "laplace",
                "supported_queries": ["count", "sum", "mean", "median", "histogram"],
                "privacy_levels": ["minimal", "standard", "high", "maximum"]
            },
            "vault": {
                "encryption": "fernet_symmetric",
                "key_derivation": "pbkdf2_sha256",
                "client_side": True
            },
            "budget_management": {
                "daily_limits": True,
                "total_limits": True,
                "automatic_reset": True
            }
        }
    
    @app.get("/api/privacy/compliance")
    async def privacy_compliance():
        """Get privacy compliance information."""
        return {
            "frameworks": {
                "differential_privacy": {
                    "standard": "epsilon_delta_dp",
                    "composition": "sequential",
                    "noise_mechanism": "laplace"
                },
                "encryption": {
                    "algorithm": "aes_256_cbc",
                    "key_derivation": "pbkdf2_sha256_600k_iterations",
                    "authentication": "hmac_sha256"
                }
            },
            "certifications": {
                "differential_privacy": "mathematically_proven",
                "encryption": "fips_140_2_level_1_compatible"
            },
            "audit_trail": {
                "budget_consumption": "complete",
                "access_logging": "comprehensive",
                "key_rotation": "supported"
            }
        }
