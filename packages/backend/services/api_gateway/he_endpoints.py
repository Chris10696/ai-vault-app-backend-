"""
FastAPI endpoints for homomorphic encryption inference
Provides REST API for encrypted AI inference operations
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np

try:
    from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field, validator
    import tenseal as ts
except ImportError as e:
    raise ImportError(f"FastAPI or TenSEAL not installed: {e}")

from ..execution.he_inference_engine import HEInferenceEngine
from ..execution.inference_pipeline import AdvancedInferencePipeline, InferenceMode
from ...core.crypto.seal_manager import SealContextManager
from ...core.crypto.key_manager import AdvancedKeyManager
from ...core.security.zero_trust import SecurityContext, get_security_context
from ...shared.types.crypto import InferenceRequest, InferenceResult, SecurityPolicy

logger = logging.getLogger(__name__)

# Request/Response Models
class ModelUploadRequest(BaseModel):
    """Request model for uploading and encrypting AI models"""
    model_id: str = Field(..., description="Unique identifier for the model")
    model_type: str = Field(..., description="Type of model (e.g., 'neural_network', 'linear')")
    architecture: Dict[str, Any] = Field(..., description="Model architecture specification")
    security_level: str = Field("high_security", description="Security level for encryption")
    description: Optional[str] = Field(None, description="Model description")
    
    @validator('model_id')
    def validate_model_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Model ID cannot be empty")
        return v.strip()
    
    @validator('security_level')
    def validate_security_level(cls, v):
        allowed_levels = ['high_security', 'medium_security', 'performance']
        if v not in allowed_levels:
            raise ValueError(f"Security level must be one of: {allowed_levels}")
        return v

class PipelineConfigRequest(BaseModel):
    """Request model for creating pipeline configurations"""
    config_id: str = Field(..., description="Unique identifier for pipeline config")
    model_id: str = Field(..., description="Model to use for inference")
    security_level: str = Field("high_security", description="Security level")
    inference_mode: str = Field("encrypted_both", description="Inference mode")
    max_batch_size: int = Field(32, description="Maximum batch size")
    timeout_seconds: int = Field(300, description="Timeout in seconds")
    enable_preprocessing: bool = Field(True, description="Enable input preprocessing")
    enable_postprocessing: bool = Field(True, description="Enable output postprocessing")
    cache_results: bool = Field(True, description="Cache inference results")
    security_policies: List[Dict[str, Any]] = Field([], description="Security policies")
    
    @validator('inference_mode')
    def validate_inference_mode(cls, v):
        allowed_modes = ['encrypted_both', 'plaintext_encrypted', 'encrypted_plaintext', 'benchmark']
        if v not in allowed_modes:
            raise ValueError(f"Inference mode must be one of: {allowed_modes}")
        return v

class InferenceRequestModel(BaseModel):
    """Request model for inference operations"""
    config_id: str = Field(..., description="Pipeline configuration ID")
    input_data: List[float] = Field(..., description="Input data for inference")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional request metadata")
    return_encrypted: bool = Field(False, description="Return encrypted output")
    
    @validator('input_data')
    def validate_input_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Input data cannot be empty")
        return v

class InferenceResponseModel(BaseModel):
    """Response model for inference operations"""
    execution_id: str = Field(..., description="Execution identifier")
    model_id: str = Field(..., description="Model used for inference")
    success: bool = Field(..., description="Whether inference succeeded")
    inference_time: float = Field(..., description="Inference time in seconds")
    total_time: float = Field(..., description="Total processing time")
    output_data: Optional[List[float]] = Field(None, description="Decrypted output data")
    encrypted_output_available: bool = Field(False, description="Whether encrypted output is available")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(..., description="Response timestamp")

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_id: str
    model_type: str
    security_level: str
    input_shape: List[int]
    output_shape: List[int]
    created_at: datetime
    performance_metrics: Dict[str, float]
    description: Optional[str]

class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status"""
    execution_id: str
    current_stage: str
    started_at: datetime
    completed_at: Optional[datetime]
    stage_timings: Dict[str, float]
    success: bool
    error_message: Optional[str]

class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    loaded_models: int
    active_sessions: int
    total_requests: int
    success_rate: float
    average_inference_time: float
    cache_hit_rate: float
    performance_metrics: Dict[str, Dict[str, float]]

# Security dependency
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> SecurityContext:
    """Extract and validate security context from request"""
    # In production, implement proper JWT validation
    return get_security_context(credentials.credentials)

# Create router
he_router = APIRouter(prefix="/he", tags=["Homomorphic Encryption"])

# Global instances (in production, use dependency injection)
_he_engine: Optional[HEInferenceEngine] = None
_pipeline: Optional[AdvancedInferencePipeline] = None

def get_he_engine() -> HEInferenceEngine:
    """Get or create HE inference engine"""
    global _he_engine
    if _he_engine is None:
        # This would be properly injected in production
        security_context = SecurityContext.create_default()
        seal_manager = SealContextManager(security_context)
        key_manager = AdvancedKeyManager(security_context)
        _he_engine = HEInferenceEngine(security_context, seal_manager, key_manager)
    return _he_engine

def get_pipeline() -> AdvancedInferencePipeline:
    """Get or create inference pipeline"""
    global _pipeline
    if _pipeline is None:
        security_context = SecurityContext.create_default()
        he_engine = get_he_engine()
        _pipeline = AdvancedInferencePipeline(security_context, he_engine)
    return _pipeline

@he_router.post("/models/upload", response_model=Dict[str, str])
async def upload_and_encrypt_model(
    request: ModelUploadRequest,
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    security_context: SecurityContext = Depends(get_current_user),
    he_engine: HEInferenceEngine = Depends(get_he_engine)
):
    """
    Upload and encrypt an AI model for homomorphic inference
    """
    try:
        logger.info(f"Uploading model: {request.model_id}")
        
        # Validate file format
        if not model_file.filename.endswith(('.pth', '.pt', '.pkl')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Use .pth, .pt, or .pkl"
            )
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{request.model_id}_{model_file.filename}"
        with open(temp_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
        
        # Encrypt model in background
        background_tasks.add_task(
            _encrypt_model_background,
            he_engine,
            temp_path,
            request.model_id,
            request.architecture,
            request.security_level
        )
        
        return {
            "message": "Model upload initiated",
            "model_id": request.model_id,
            "status": "encrypting"
        }
        
    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model upload failed: {str(e)}"
        )

async def _encrypt_model_background(
    he_engine: HEInferenceEngine,
    model_path: str,
    model_id: str,
    architecture: Dict[str, Any],
    security_level: str
):
    """Background task to encrypt uploaded model"""
    try:
        he_engine.load_and_encrypt_model(
            model_path,
            model_id,
            architecture,
            security_level
        )
        
        # Clean up temporary file
        os.remove(model_path)
        
        logger.info(f"Model {model_id} encrypted successfully")
        
    except Exception as e:
        logger.error(f"Background model encryption failed: {e}")
        # Clean up on failure
        if os.path.exists(model_path):
            os.remove(model_path)

@he_router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model_info(
    model_id: str,
    security_context: SecurityContext = Depends(get_current_user),
    he_engine: HEInferenceEngine = Depends(get_he_engine)
):
    """Get information about an encrypted model"""
    try:
        if model_id not in he_engine.encrypted_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        encrypted_model = he_engine.encrypted_models[model_id]
        metadata = encrypted_model.model_metadata
        
        return ModelInfoResponse(
            model_id=metadata.model_id,
            model_type=metadata.model_type,
            security_level=metadata.security_level,
            input_shape=metadata.input_shape,
            output_shape=metadata.output_shape,
            created_at=metadata.created_at,
            performance_metrics=encrypted_model.performance_metrics,
            description=getattr(metadata, 'description', None)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get model info failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )

@he_router.post("/pipeline/config", response_model=Dict[str, str])
async def create_pipeline_config(
    request: PipelineConfigRequest,
    security_context: SecurityContext = Depends(get_current_user),
    pipeline: AdvancedInferencePipeline = Depends(get_pipeline)
):
    """Create a new inference pipeline configuration"""
    try:
        # Convert inference mode string to enum
        inference_mode = InferenceMode(request.inference_mode)
        
        # Convert security policies
        security_policies = [
            SecurityPolicy(**policy) for policy in request.security_policies
        ]
        
        config_id = pipeline.create_pipeline_config(
            config_id=request.config_id,
            model_id=request.model_id,
            security_level=request.security_level,
            inference_mode=inference_mode,
            max_batch_size=request.max_batch_size,
            timeout_seconds=request.timeout_seconds,
            enable_preprocessing=request.enable_preprocessing,
            enable_postprocessing=request.enable_postprocessing,
            cache_results=request.cache_results,
            security_policies=security_policies
        )
        
        return {
            "message": "Pipeline configuration created",
            "config_id": config_id
        }
        
    except Exception as e:
        logger.error(f"Pipeline config creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pipeline configuration: {str(e)}"
        )

@he_router.post("/inference", response_model=InferenceResponseModel)
async def run_inference(
    request: InferenceRequestModel,
    security_context: SecurityContext = Depends(get_current_user),
    pipeline: AdvancedInferencePipeline = Depends(get_pipeline)
):
    """Run encrypted inference on input data"""
    try:
        logger.info(f"Running inference with config: {request.config_id}")
        
        # Convert input data to numpy array
        input_array = np.array(request.input_data, dtype=np.float64)
        
        # Execute pipeline
        result = await pipeline.execute_pipeline(
            config_id=request.config_id,
            input_data=input_array,
            request_metadata=request.metadata
        )
        
        # Prepare response
        output_data = None
        encrypted_output_available = False
        
        if result.success:
            if hasattr(result, 'decrypted_output') and result.decrypted_output is not None:
                output_data = result.decrypted_output.tolist()
            elif not request.return_encrypted and result.encrypted_output is not None:
                # Decrypt output for response
                try:
                    session_id = pipeline.he_engine.create_inference_session(
                        result.model_id, "high_security"
                    )
                    decrypted = pipeline.he_engine.decrypt_result(result, session_id)
                    output_data = decrypted.tolist()
                    pipeline.he_engine.close_session(session_id)
                except Exception as e:
                    logger.warning(f"Failed to decrypt output for response: {e}")
            
            if result.encrypted_output is not None:
                encrypted_output_available = True
        
        return InferenceResponseModel(
            execution_id=getattr(result, 'execution_id', 'unknown'),
            model_id=result.model_id,
            success=result.success,
            inference_time=result.inference_time,
            total_time=result.total_time,
            output_data=output_data,
            encrypted_output_available=encrypted_output_available,
            error_message=result.error_message,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@he_router.get("/inference/status/{execution_id}", response_model=PipelineStatusResponse)
async def get_inference_status(
    execution_id: str,
    security_context: SecurityContext = Depends(get_current_user),
    pipeline: AdvancedInferencePipeline = Depends(get_pipeline)
):
    """Get status of pipeline execution"""
    try:
        status_info = pipeline.get_pipeline_status(execution_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found"
            )
        
        return PipelineStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get status failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution status"
        )

@he_router.get("/system/stats", response_model=SystemStatsResponse)
async def get_system_statistics(
    security_context: SecurityContext = Depends(get_current_user),
    pipeline: AdvancedInferencePipeline = Depends(get_pipeline)
):
    """Get comprehensive system statistics"""
    try:
        stats = pipeline.get_pipeline_statistics()
        
        # Calculate additional metrics
        success_rate = stats.get('success_rate', 0.0)
        
        # Get average inference time
        he_stats = stats.get('he_engine_stats', {})
        perf_metrics = he_stats.get('performance_metrics', {})
        avg_inference_time = 0.0
        
        if 'inference_time' in perf_metrics:
            avg_inference_time = perf_metrics['inference_time'].get('avg', 0.0)
        
        # Calculate cache hit rate (placeholder)
        cache_hit_rate = 0.85  # This would be calculated from actual cache statistics
        
        return SystemStatsResponse(
            loaded_models=stats.get('total_executions', 0),
            active_sessions=stats.get('active_executions', 0),
            total_requests=stats.get('total_executions', 0),
            success_rate=success_rate,
            average_inference_time=avg_inference_time,
            cache_hit_rate=cache_hit_rate,
            performance_metrics=perf_metrics
        )
        
    except Exception as e:
        logger.error(f"Get system stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )

@he_router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    security_context: SecurityContext = Depends(get_current_user),
    he_engine: HEInferenceEngine = Depends(get_he_engine)
):
    """Delete an encrypted model"""
    try:
        if model_id not in he_engine.encrypted_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Remove from memory
        del he_engine.encrypted_models[model_id]
        
        # Clean up files
        model_file = f"{he_engine.model_store_path}/encrypted/{model_id}.pkl"
        metadata_file = f"{he_engine.model_store_path}/metadata/{model_id}.json"
        
        for file_path in [model_file, metadata_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        logger.info(f"Model {model_id} deleted successfully")
        
        return {"message": f"Model {model_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )

@he_router.post("/system/cleanup")
async def system_cleanup(
    max_age_hours: int = 24,
    security_context: SecurityContext = Depends(get_current_user),
    pipeline: AdvancedInferencePipeline = Depends(get_pipeline)
):
    """Perform system cleanup of old sessions and cache"""
    try:
        # Clean up old pipeline executions
        pipeline.cleanup_old_executions(max_age_hours)
        
        # Clean up HE engine sessions
        pipeline.he_engine.cleanup_inactive_sessions(max_age_hours)
        
        logger.info("System cleanup completed")
        
        return {"message": "System cleanup completed successfully"}
        
    except Exception as e:
        logger.error(f"System cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System cleanup failed: {str(e)}"
        )
