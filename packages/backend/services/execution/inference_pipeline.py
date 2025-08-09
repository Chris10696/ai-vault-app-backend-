"""
Complete AI Inference Pipeline with Homomorphic Encryption
Provides high-level interface for encrypted AI inference
"""

import os
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

try:
    import tenseal as ts
    from fastapi import HTTPException, status
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}")

from .he_inference_engine import HEInferenceEngine, SecureNeuralNetwork
from ...core.crypto.seal_manager import SealContextManager
from ...core.crypto.key_manager import AdvancedKeyManager, KeyType
from ...core.security.zero_trust import SecurityContext, TrustLevel
from ...shared.types.crypto import (
    InferenceRequest, 
    InferenceResult, 
    ModelDeployment,
    PipelineStatus,
    SecurityPolicy
)

logger = logging.getLogger(__name__)

class InferenceMode(Enum):
    """Inference execution modes"""
    ENCRYPTED_INPUT_ENCRYPTED_OUTPUT = "encrypted_both"
    PLAINTEXT_INPUT_ENCRYPTED_OUTPUT = "plaintext_encrypted"  
    ENCRYPTED_INPUT_PLAINTEXT_OUTPUT = "encrypted_plaintext"
    BENCHMARK = "benchmark"

class PipelineStage(Enum):
    """Pipeline execution stages"""
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    ENCRYPTION = "encryption"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"
    DECRYPTION = "decryption"
    COMPLETE = "complete"

@dataclass
class InferencePipelineConfig:
    """Configuration for inference pipeline"""
    model_id: str
    security_level: str
    inference_mode: InferenceMode
    max_batch_size: int
    timeout_seconds: int
    enable_preprocessing: bool
    enable_postprocessing: bool
    cache_results: bool
    security_policies: List[SecurityPolicy]

@dataclass
class PipelineExecution:
    """Track pipeline execution"""
    execution_id: str
    pipeline_config: InferencePipelineConfig
    current_stage: PipelineStage
    started_at: datetime
    completed_at: Optional[datetime]
    stage_timings: Dict[str, float]
    success: bool
    error_message: Optional[str]
    results_cache_key: Optional[str]

class AdvancedInferencePipeline:
    """
    Advanced inference pipeline with comprehensive HE support
    Features:
    - Multiple inference modes
    - Batch processing
    - Preprocessing/postprocessing
    - Result caching
    - Performance optimization
    - Security policy enforcement
    """
    
    def __init__(
        self,
        security_context: SecurityContext,
        he_engine: HEInferenceEngine,
        config_path: Optional[str] = None
    ):
        self.security_context = security_context
        self.he_engine = he_engine
        self.config_path = config_path or os.getenv("PIPELINE_CONFIG_PATH", "pipeline_configs")
        
        # Pipeline state management
        self.active_executions: Dict[str, PipelineExecution] = {}
        self.pipeline_configs: Dict[str, InferencePipelineConfig] = {}
        self.model_deployments: Dict[str, ModelDeployment] = {}
        
        # Performance and caching
        self.results_cache: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # Threading and async support
        self._lock = threading.RLock()
        self._executor = None  # Will be initialized when needed
        
        # Load configurations
        self._load_pipeline_configs()
        
        logger.info("Advanced Inference Pipeline initialized")
    
    def _load_pipeline_configs(self):
        """Load pipeline configurations from disk"""
        try:
            os.makedirs(self.config_path, exist_ok=True)
            config_file = f"{self.config_path}/pipeline_configs.json"
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    configs_data = json.load(f)
                
                for config_id, config_data in configs_data.items():
                    config = InferencePipelineConfig(
                        model_id=config_data['model_id'],
                        security_level=config_data['security_level'],
                        inference_mode=InferenceMode(config_data['inference_mode']),
                        max_batch_size=config_data['max_batch_size'],
                        timeout_seconds=config_data['timeout_seconds'],
                        enable_preprocessing=config_data['enable_preprocessing'],
                        enable_postprocessing=config_data['enable_postprocessing'],
                        cache_results=config_data['cache_results'],
                        security_policies=[
                            SecurityPolicy(**policy) for policy in config_data['security_policies']
                        ]
                    )
                    self.pipeline_configs[config_id] = config
                
                logger.info(f"Loaded {len(self.pipeline_configs)} pipeline configurations")
        
        except Exception as e:
            logger.warning(f"Failed to load pipeline configurations: {e}")
    
    def create_pipeline_config(
        self,
        config_id: str,
        model_id: str,
        security_level: str = "high_security",
        inference_mode: InferenceMode = InferenceMode.ENCRYPTED_INPUT_ENCRYPTED_OUTPUT,
        **kwargs
    ) -> str:
        """Create a new pipeline configuration"""
        with self._lock:
            try:
                # Validate model exists
                if not self._validate_model_exists(model_id):
                    raise ValueError(f"Model {model_id} not found")
                
                # Create configuration
                config = InferencePipelineConfig(
                    model_id=model_id,
                    security_level=security_level,
                    inference_mode=inference_mode,
                    max_batch_size=kwargs.get('max_batch_size', 32),
                    timeout_seconds=kwargs.get('timeout_seconds', 300),
                    enable_preprocessing=kwargs.get('enable_preprocessing', True),
                    enable_postprocessing=kwargs.get('enable_postprocessing', True),
                    cache_results=kwargs.get('cache_results', True),
                    security_policies=kwargs.get('security_policies', [])
                )
                
                self.pipeline_configs[config_id] = config
                self._save_pipeline_configs()
                
                logger.info(f"Created pipeline configuration: {config_id}")
                return config_id
                
            except Exception as e:
                logger.error(f"Failed to create pipeline config {config_id}: {e}")
                raise
    
    async def execute_pipeline(
        self,
        config_id: str,
        input_data: Union[List[float], np.ndarray, ts.CKKSVector],
        request_metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """
        Execute complete inference pipeline asynchronously
        """
        execution_id = f"exec_{config_id}_{int(time.time())}_{os.urandom(4).hex()}"
        
        try:
            # Get pipeline configuration
            if config_id not in self.pipeline_configs:
                raise ValueError(f"Pipeline configuration {config_id} not found")
            
            config = self.pipeline_configs[config_id]
            
            # Create pipeline execution tracker
            execution = PipelineExecution(
                execution_id=execution_id,
                pipeline_config=config,
                current_stage=PipelineStage.VALIDATION,
                started_at=datetime.utcnow(),
                completed_at=None,
                stage_timings={},
                success=False,
                error_message=None,
                results_cache_key=None
            )
            
            self.active_executions[execution_id] = execution
            
            # Execute pipeline stages
            result = await self._execute_pipeline_stages(execution, input_data, request_metadata)
            
            # Mark as complete
            execution.completed_at = datetime.utcnow()
            execution.success = result.success
            
            logger.info(f"Pipeline execution {execution_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution {execution_id} failed: {e}")
            
            if execution_id in self.active_executions:
                self.active_executions[execution_id].error_message = str(e)
            
            # Create error result
            return InferenceResult(
                session_id="",
                model_id=config.model_id if 'config' in locals() else "unknown",
                encrypted_output=None,
                inference_time=0.0,
                total_time=0.0,
                encryption_time=0.0,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    async def _execute_pipeline_stages(
        self,
        execution: PipelineExecution,
        input_data: Union[List[float], np.ndarray, ts.CKKSVector],
        request_metadata: Optional[Dict[str, Any]]
    ) -> InferenceResult:
        """Execute all pipeline stages"""
        
        config = execution.pipeline_config
        
        # Stage 1: Validation
        execution.current_stage = PipelineStage.VALIDATION
        stage_start = time.time()
        
        await self._validate_request(config, input_data, request_metadata)
        
        execution.stage_timings['validation'] = time.time() - stage_start
        
        # Stage 2: Preprocessing (if enabled)
        if config.enable_preprocessing:
            execution.current_stage = PipelineStage.PREPROCESSING
            stage_start = time.time()
            
            processed_input = await self._preprocess_input(config, input_data)
            
            execution.stage_timings['preprocessing'] = time.time() - stage_start
        else:
            processed_input = input_data
        
        # Stage 3: Encryption (if needed)
        execution.current_stage = PipelineStage.ENCRYPTION
        stage_start = time.time()
        
        encrypted_input = await self._handle_input_encryption(config, processed_input)
        
        execution.stage_timings['encryption'] = time.time() - stage_start
        
        # Stage 4: Inference
        execution.current_stage = PipelineStage.INFERENCE
        stage_start = time.time()
        
        inference_result = await self._execute_inference(config, encrypted_input)
        
        execution.stage_timings['inference'] = time.time() - stage_start
        
        # Stage 5: Postprocessing (if enabled)
        if config.enable_postprocessing:
            execution.current_stage = PipelineStage.POSTPROCESSING
            stage_start = time.time()
            
            processed_result = await self._postprocess_result(config, inference_result)
            
            execution.stage_timings['postprocessing'] = time.time() - stage_start
        else:
            processed_result = inference_result
        
        # Stage 6: Decryption (if needed)
        execution.current_stage = PipelineStage.DECRYPTION
        stage_start = time.time()
        
        final_result = await self._handle_output_decryption(config, processed_result)
        
        execution.stage_timings['decryption'] = time.time() - stage_start
        
        # Mark as complete
        execution.current_stage = PipelineStage.COMPLETE
        
        # Cache results if enabled
        if config.cache_results:
            await self._cache_result(execution, final_result)
        
        return final_result
    
    async def _validate_request(
        self,
        config: InferencePipelineConfig,
        input_data: Union[List[float], np.ndarray, ts.CKKSVector],
        request_metadata: Optional[Dict[str, Any]]
    ):
        """Validate inference request"""
        
        # Check security policies
        for policy in config.security_policies:
            if not await self._enforce_security_policy(policy, request_metadata):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Security policy violation: {policy.name}"
                )
        
        # Validate input data format
        if isinstance(input_data, (list, np.ndarray)):
            if isinstance(input_data, list):
                input_array = np.array(input_data)
            else:
                input_array = input_data
            
            # Check batch size
            if len(input_array) > config.max_batch_size:
                raise ValueError(f"Batch size {len(input_array)} exceeds limit {config.max_batch_size}")
        
        # Validate model is available
        if not self._validate_model_exists(config.model_id):
            raise ValueError(f"Model {config.model_id} not available")
    
    async def _preprocess_input(
        self,
        config: InferencePipelineConfig,
        input_data: Union[List[float], np.ndarray, ts.CKKSVector]
    ) -> Union[List[float], np.ndarray, ts.CKKSVector]:
        """Preprocess input data"""
        
        if isinstance(input_data, ts.CKKSVector):
            # Already encrypted, limited preprocessing options
            return input_data
        
        # Convert to numpy array for processing
        if isinstance(input_data, list):
            data_array = np.array(input_data, dtype=np.float64)
        else:
            data_array = input_data.astype(np.float64)
        
        # Apply preprocessing based on model requirements
        # This would be customized based on specific model needs
        
        # Normalization
        if data_array.std() > 0:
            data_array = (data_array - data_array.mean()) / data_array.std()
        
        # Padding or truncation if needed
        target_size = self._get_model_input_size(config.model_id)
        if target_size and len(data_array) != target_size:
            if len(data_array) > target_size:
                data_array = data_array[:target_size]
            else:
                padding = np.zeros(target_size - len(data_array))
                data_array = np.concatenate([data_array, padding])
        
        return data_array
    
    async def _handle_input_encryption(
        self,
        config: InferencePipelineConfig,
        input_data: Union[List[float], np.ndarray, ts.CKKSVector]
    ) -> ts.CKKSVector:
        """Handle input encryption based on inference mode"""
        
        if isinstance(input_data, ts.CKKSVector):
            # Already encrypted
            return input_data
        
        if config.inference_mode in [
            InferenceMode.ENCRYPTED_INPUT_ENCRYPTED_OUTPUT,
            InferenceMode.ENCRYPTED_INPUT_PLAINTEXT_OUTPUT
        ]:
            # Need to encrypt input
            session_id = self.he_engine.create_inference_session(
                config.model_id, config.security_level
            )
            
            op_manager = self.he_engine.op_managers[config.security_level]
            
            if isinstance(input_data, np.ndarray):
                input_data = input_data.tolist()
            
            encrypted_input = op_manager.encrypt_vector(input_data, config.security_level)
            return encrypted_input
        
        else:
            # Return as-is for plaintext modes
            return input_data
    
    async def _execute_inference(
        self,
        config: InferencePipelineConfig,
        input_data: Union[List[float], np.ndarray, ts.CKKSVector]
    ) -> InferenceResult:
        """Execute the actual inference"""
        
        # Create or get inference session
        session_id = self.he_engine.create_inference_session(
            config.model_id, config.security_level
        )
        
        try:
            # Run inference
            result = self.he_engine.run_encrypted_inference(session_id, input_data)
            return result
            
        finally:
            # Clean up session if not needed
            self.he_engine.close_session(session_id)
    
    async def _postprocess_result(
        self,
        config: InferencePipelineConfig,
        inference_result: InferenceResult
    ) -> InferenceResult:
        """Postprocess inference results"""
        
        if not inference_result.success:
            return inference_result
        
        # Apply postprocessing based on model requirements
        # This is a placeholder for model-specific postprocessing
        
        return inference_result
    
    async def _handle_output_decryption(
        self,
        config: InferencePipelineConfig,
        inference_result: InferenceResult
    ) -> InferenceResult:
        """Handle output decryption based on inference mode"""
        
        if not inference_result.success:
            return inference_result
        
        if config.inference_mode in [
            InferenceMode.ENCRYPTED_INPUT_PLAINTEXT_OUTPUT,
            InferenceMode.PLAINTEXT_INPUT_ENCRYPTED_OUTPUT
        ]:
            # Need to decrypt output
            if inference_result.encrypted_output is not None:
                session_id = self.he_engine.create_inference_session(
                    config.model_id, config.security_level
                )
                
                try:
                    decrypted_output = self.he_engine.decrypt_result(
                        inference_result, session_id
                    )
                    
                    # Create new result with decrypted output
                    inference_result.decrypted_output = decrypted_output
                    
                finally:
                    self.he_engine.close_session(session_id)
        
        return inference_result
    
    async def _cache_result(
        self,
        execution: PipelineExecution,
        result: InferenceResult
    ):
        """Cache inference result"""
        try:
            cache_key = f"result_{execution.execution_id}_{hash(str(result.timestamp))}"
            
            # Store in memory cache (in production, use Redis or similar)
            self.results_cache[cache_key] = {
                'result': asdict(result),
                'execution_info': asdict(execution),
                'cached_at': datetime.utcnow()
            }
            
            execution.results_cache_key = cache_key
            
            # Limit cache size
            if len(self.results_cache) > 1000:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.results_cache.keys(),
                    key=lambda k: self.results_cache[k]['cached_at']
                )
                for key in sorted_keys[:100]:
                    del self.results_cache[key]
        
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    async def _enforce_security_policy(
        self,
        policy: SecurityPolicy,
        request_metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Enforce security policy"""
        
        # Check trust level requirements
        current_trust = self.security_context.get_trust_level()
        if policy.min_trust_level and current_trust.value < policy.min_trust_level.value:
            return False
        
        # Check rate limits
        if policy.rate_limit_per_hour:
            # This would check against a rate limiting store
            pass
        
        # Check IP restrictions
        if policy.allowed_ips and request_metadata:
            client_ip = request_metadata.get('client_ip')
            if client_ip and client_ip not in policy.allowed_ips:
                return False
        
        return True
    
    def _validate_model_exists(self, model_id: str) -> bool:
        """Validate that model exists and is available"""
        return model_id in self.he_engine.encrypted_models
    
    def _get_model_input_size(self, model_id: str) -> Optional[int]:
        """Get expected input size for model"""
        if model_id in self.he_engine.encrypted_models:
            model = self.he_engine.encrypted_models[model_id]
            input_shape = model.model_metadata.input_shape
            if input_shape:
                return np.prod(input_shape)
        return None
    
    def _save_pipeline_configs(self):
        """Save pipeline configurations to disk"""
        try:
            config_file = f"{self.config_path}/pipeline_configs.json"
            
            configs_data = {}
            for config_id, config in self.pipeline_configs.items():
                configs_data[config_id] = {
                    'model_id': config.model_id,
                    'security_level': config.security_level,
                    'inference_mode': config.inference_mode.value,
                    'max_batch_size': config.max_batch_size,
                    'timeout_seconds': config.timeout_seconds,
                    'enable_preprocessing': config.enable_preprocessing,
                    'enable_postprocessing': config.enable_postprocessing,
                    'cache_results': config.cache_results,
                    'security_policies': [asdict(policy) for policy in config.security_policies]
                }
            
            with open(config_file, 'w') as f:
                json.dump(configs_data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save pipeline configs: {e}")
    
    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of pipeline execution"""
        if execution_id not in self.active_executions:
            return None
        
        execution = self.active_executions[execution_id]
        
        return {
            'execution_id': execution_id,
            'current_stage': execution.current_stage.value,
            'started_at': execution.started_at.isoformat(),
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'stage_timings': execution.stage_timings,
            'success': execution.success,
            'error_message': execution.error_message
        }
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        with self._lock:
            active_count = len([e for e in self.active_executions.values() 
                               if e.current_stage != PipelineStage.COMPLETE])
            
            completed_count = len([e for e in self.active_executions.values() 
                                  if e.current_stage == PipelineStage.COMPLETE])
            
            success_count = len([e for e in self.active_executions.values() 
                                if e.success])
            
            # Calculate average stage timings
            stage_averages = {}
            for execution in self.active_executions.values():
                for stage, timing in execution.stage_timings.items():
                    if stage not in stage_averages:
                        stage_averages[stage] = []
                    stage_averages[stage].append(timing)
            
            for stage in stage_averages:
                stage_averages[stage] = {
                    'avg': np.mean(stage_averages[stage]),
                    'min': np.min(stage_averages[stage]),
                    'max': np.max(stage_averages[stage])
                }
            
            return {
                'total_executions': len(self.active_executions),
                'active_executions': active_count,
                'completed_executions': completed_count,
                'success_rate': success_count / len(self.active_executions) if self.active_executions else 0,
                'pipeline_configs': len(self.pipeline_configs),
                'cached_results': len(self.results_cache),
                'stage_performance': stage_averages,
                'he_engine_stats': self.he_engine.get_inference_statistics()
            }
    
    def cleanup_old_executions(self, max_age_hours: int = 24):
        """Clean up old pipeline executions"""
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            executions_to_remove = []
            for execution_id, execution in self.active_executions.items():
                if (execution.completed_at and execution.completed_at < cutoff_time) or \
                   (not execution.completed_at and execution.started_at < cutoff_time):
                    executions_to_remove.append(execution_id)
            
            for execution_id in executions_to_remove:
                del self.active_executions[execution_id]
            
            logger.info(f"Cleaned up {len(executions_to_remove)} old pipeline executions")
