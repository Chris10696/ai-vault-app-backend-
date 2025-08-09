"""
Homomorphic Encryption AI Inference Engine
Enables secure AI model execution on encrypted data
"""

import os
import json
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle

try:
    import tenseal as ts
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError(f"Required ML libraries not installed: {e}")

from ...core.crypto.seal_manager import SealContextManager, HomomorphicOperationManager
from ...core.crypto.key_manager import AdvancedKeyManager, KeyType
from ...core.security.zero_trust import SecurityContext
from ...shared.types.crypto import InferenceRequest, InferenceResult, ModelMetadata

logger = logging.getLogger(__name__)

@dataclass
class EncryptedModel:
    """Encrypted AI model representation"""
    model_id: str
    model_type: str
    encrypted_weights: Dict[str, ts.CKKSVector]
    model_metadata: ModelMetadata
    encryption_context: str
    created_at: datetime
    performance_metrics: Dict[str, float]

@dataclass
class InferenceSession:
    """Secure inference session"""
    session_id: str
    model_id: str
    encryption_context: str
    created_at: datetime
    active: bool
    request_count: int
    total_inference_time: float

class SecureNeuralNetwork:
    """
    Neural network designed for homomorphic encryption
    Supports basic operations: linear layers, polynomial activations
    """
    
    def __init__(self, architecture: Dict[str, Any]):
        self.architecture = architecture
        self.layers = []
        self.encrypted_weights = {}
        self.encrypted_biases = {}
        
        # Build network architecture
        self._build_network()
    
    def _build_network(self):
        """Build network layers from architecture specification"""
        layers_config = self.architecture.get('layers', [])
        
        for i, layer_config in enumerate(layers_config):
            layer_type = layer_config.get('type')
            
            if layer_type == 'linear':
                layer = self._create_linear_layer(layer_config, i)
                self.layers.append(layer)
            
            elif layer_type == 'polynomial':
                layer = self._create_polynomial_layer(layer_config, i)
                self.layers.append(layer)
            
            elif layer_type == 'pooling':
                layer = self._create_pooling_layer(layer_config, i)
                self.layers.append(layer)
            
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
    
    def _create_linear_layer(self, config: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
        """Create linear layer configuration"""
        return {
            'type': 'linear',
            'index': layer_idx,
            'input_size': config.get('input_size'),
            'output_size': config.get('output_size'),
            'use_bias': config.get('use_bias', True),
            'weight_key': f"layer_{layer_idx}_weight",
            'bias_key': f"layer_{layer_idx}_bias" if config.get('use_bias') else None
        }
    
    def _create_polynomial_layer(self, config: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
        """Create polynomial activation layer"""
        return {
            'type': 'polynomial',
            'index': layer_idx,
            'degree': config.get('degree', 2),
            'coefficients': config.get('coefficients', [0, 1, 0.25])  # Default: x + 0.25*x^2
        }
    
    def _create_pooling_layer(self, config: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
        """Create pooling layer (average pooling for HE compatibility)"""
        return {
            'type': 'pooling',
            'index': layer_idx,
            'kernel_size': config.get('kernel_size', 2),
            'stride': config.get('stride', 2),
            'pool_type': 'average'  # Only average pooling is HE-friendly
        }
    
    def encrypt_model(
        self, 
        model_weights: Dict[str, np.ndarray], 
        op_manager: HomomorphicOperationManager,
        context_name: str = "high_security"
    ):
        """Encrypt model weights for homomorphic computation"""
        try:
            logger.info("Starting model encryption...")
            
            for layer in self.layers:
                if layer['type'] == 'linear':
                    weight_key = layer['weight_key']
                    bias_key = layer['bias_key']
                    
                    # Encrypt weights
                    if weight_key in model_weights:
                        weight_matrix = model_weights[weight_key]
                        
                        # Flatten and encrypt weight matrix
                        flattened_weights = weight_matrix.flatten().tolist()
                        encrypted_weights = op_manager.encrypt_vector(
                            flattened_weights, context_name
                        )
                        self.encrypted_weights[weight_key] = encrypted_weights
                        
                        logger.debug(f"Encrypted weights for layer {layer['index']}: {weight_matrix.shape}")
                    
                    # Encrypt biases
                    if bias_key and bias_key in model_weights:
                        bias_vector = model_weights[bias_key]
                        encrypted_bias = op_manager.encrypt_vector(
                            bias_vector.tolist(), context_name
                        )
                        self.encrypted_biases[bias_key] = encrypted_bias
                        
                        logger.debug(f"Encrypted bias for layer {layer['index']}: {bias_vector.shape}")
            
            logger.info("Model encryption completed successfully")
            
        except Exception as e:
            logger.error(f"Model encryption failed: {e}")
            raise
    
    def forward_encrypted(
        self, 
        encrypted_input: ts.CKKSVector, 
        op_manager: HomomorphicOperationManager
    ) -> ts.CKKSVector:
        """Perform forward pass on encrypted data"""
        try:
            current_output = encrypted_input
            
            for layer in self.layers:
                if layer['type'] == 'linear':
                    current_output = self._linear_layer_encrypted(
                        current_output, layer, op_manager
                    )
                
                elif layer['type'] == 'polynomial':
                    current_output = self._polynomial_activation_encrypted(
                        current_output, layer, op_manager
                    )
                
                elif layer['type'] == 'pooling':
                    current_output = self._pooling_layer_encrypted(
                        current_output, layer, op_manager
                    )
                
                logger.debug(f"Completed layer {layer['index']} ({layer['type']})")
            
            return current_output
            
        except Exception as e:
            logger.error(f"Encrypted forward pass failed: {e}")
            raise
    
    def _linear_layer_encrypted(
        self, 
        encrypted_input: ts.CKKSVector, 
        layer: Dict[str, Any], 
        op_manager: HomomorphicOperationManager
    ) -> ts.CKKSVector:
        """Perform encrypted linear layer computation"""
        try:
            weight_key = layer['weight_key']
            bias_key = layer['bias_key']
            
            # Get encrypted weights
            if weight_key not in self.encrypted_weights:
                raise ValueError(f"Encrypted weights not found for {weight_key}")
            
            encrypted_weights = self.encrypted_weights[weight_key]
            
            # Perform matrix multiplication (simplified for demonstration)
            # In practice, this would require more sophisticated SIMD operations
            result = op_manager.homomorphic_multiply(encrypted_input, encrypted_weights)
            
            # Add bias if present
            if bias_key and bias_key in self.encrypted_biases:
                encrypted_bias = self.encrypted_biases[bias_key]
                result = op_manager.homomorphic_add(result, encrypted_bias)
            
            return result
            
        except Exception as e:
            logger.error(f"Encrypted linear layer failed: {e}")
            raise
    
    def _polynomial_activation_encrypted(
        self, 
        encrypted_input: ts.CKKSVector, 
        layer: Dict[str, Any], 
        op_manager: HomomorphicOperationManager
    ) -> ts.CKKSVector:
        """Perform encrypted polynomial activation"""
        try:
            coefficients = layer['coefficients']
            degree = layer['degree']
            
            # Initialize result with constant term
            if len(coefficients) > 0:
                context = encrypted_input.context()
                const_data = np.full(encrypted_input.size(), coefficients[0])
                result = ts.ckks_vector(context, const_data.tolist())
            else:
                result = encrypted_input
            
            # Add polynomial terms
            current_power = encrypted_input
            
            for i in range(1, min(degree + 1, len(coefficients))):
                if coefficients[i] != 0:
                    # Create coefficient vector
                    coeff_data = np.full(encrypted_input.size(), coefficients[i])
                    coeff_vector = ts.ckks_vector(context, coeff_data.tolist())
                    
                    # Add weighted term
                    weighted_term = op_manager.homomorphic_multiply(current_power, coeff_vector)
                    result = op_manager.homomorphic_add(result, weighted_term)
                
                # Compute next power
                if i < degree:
                    current_power = op_manager.homomorphic_multiply(current_power, encrypted_input)
            
            return result
            
        except Exception as e:
            logger.error(f"Encrypted polynomial activation failed: {e}")
            raise
    
    def _pooling_layer_encrypted(
        self, 
        encrypted_input: ts.CKKSVector, 
        layer: Dict[str, Any], 
        op_manager: HomomorphicOperationManager
    ) -> ts.CKKSVector:
        """Perform encrypted pooling (average pooling)"""
        try:
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            
            # For simplicity, implement basic average pooling
            # In practice, this would require rotation operations
            
            # Create scaling factor for average
            context = encrypted_input.context()
            scale_factor = 1.0 / (kernel_size * kernel_size)
            scale_data = np.full(encrypted_input.size(), scale_factor)
            scale_vector = ts.ckks_vector(context, scale_data.tolist())
            
            # Apply scaling (simplified pooling)
            result = op_manager.homomorphic_multiply(encrypted_input, scale_vector)
            
            return result
            
        except Exception as e:
            logger.error(f"Encrypted pooling layer failed: {e}")
            raise

class HEInferenceEngine:
    """
    Main inference engine for homomorphic encryption
    Manages encrypted models and inference sessions
    """
    
    def __init__(
        self, 
        security_context: SecurityContext,
        seal_manager: SealContextManager,
        key_manager: AdvancedKeyManager,
        model_store_path: Optional[str] = None
    ):
        self.security_context = security_context
        self.seal_manager = seal_manager
        self.key_manager = key_manager
        self.model_store_path = model_store_path or os.getenv("HE_MODEL_STORE", "he_models")
        
        # Operation managers for different security levels
        self.op_managers = {
            'high_security': HomomorphicOperationManager(seal_manager),
            'medium_security': HomomorphicOperationManager(seal_manager),
            'performance': HomomorphicOperationManager(seal_manager)
        }
        
        # Model and session management
        self.encrypted_models: Dict[str, EncryptedModel] = {}
        self.active_sessions: Dict[str, InferenceSession] = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self.inference_metrics: Dict[str, List[float]] = {
            'encryption_time': [],
            'inference_time': [],
            'decryption_time': [],
            'total_time': []
        }
        
        self._setup_model_store()
    
    def _setup_model_store(self):
        """Setup model storage directory"""
        os.makedirs(self.model_store_path, exist_ok=True)
        os.makedirs(f"{self.model_store_path}/encrypted", exist_ok=True)
        os.makedirs(f"{self.model_store_path}/metadata", exist_ok=True)
    
    def load_and_encrypt_model(
        self, 
        model_path: str, 
        model_id: str,
        architecture: Dict[str, Any],
        security_level: str = "high_security"
    ) -> str:
        """
        Load a PyTorch model and encrypt it for homomorphic inference
        """
        with self._lock:
            try:
                logger.info(f"Loading and encrypting model {model_id}")
                
                # Load PyTorch model
                if model_path.endswith('.pth') or model_path.endswith('.pt'):
                    model_weights = torch.load(model_path, map_location='cpu')
                elif model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        model_weights = pickle.load(f)
                else:
                    raise ValueError("Unsupported model format. Use .pth, .pt, or .pkl")
                
                # Convert torch tensors to numpy
                if isinstance(model_weights, dict):
                    numpy_weights = {}
                    for key, tensor in model_weights.items():
                        if hasattr(tensor, 'numpy'):
                            numpy_weights[key] = tensor.detach().numpy()
                        else:
                            numpy_weights[key] = np.array(tensor)
                else:
                    raise ValueError("Model weights must be a dictionary")
                
                # Create secure neural network
                secure_nn = SecureNeuralNetwork(architecture)
                
                # Encrypt the model
                op_manager = self.op_managers[security_level]
                secure_nn.encrypt_model(numpy_weights, op_manager, security_level)
                
                # Create model metadata
                metadata = ModelMetadata(
                    model_id=model_id,
                    model_type=architecture.get('type', 'unknown'),
                    input_shape=architecture.get('input_shape', []),
                    output_shape=architecture.get('output_shape', []),
                    security_level=security_level,
                    created_at=datetime.utcnow(),
                    performance_requirements=architecture.get('performance', {})
                )
                
                # Store encrypted model
                encrypted_model = EncryptedModel(
                    model_id=model_id,
                    model_type=metadata.model_type,
                    encrypted_weights={},  # Store reference to secure_nn
                    model_metadata=metadata,
                    encryption_context=security_level,
                    created_at=datetime.utcnow(),
                    performance_metrics={}
                )
                
                # Store the secure neural network reference
                encrypted_model.secure_nn = secure_nn
                
                self.encrypted_models[model_id] = encrypted_model
                
                # Persist to disk
                self._save_encrypted_model(encrypted_model)
                
                logger.info(f"Model {model_id} encrypted and loaded successfully")
                return model_id
                
            except Exception as e:
                logger.error(f"Failed to load and encrypt model {model_id}: {e}")
                raise
    
    def create_inference_session(
        self, 
        model_id: str, 
        security_level: str = "high_security"
    ) -> str:
        """Create a new inference session"""
        with self._lock:
            try:
                if model_id not in self.encrypted_models:
                    raise ValueError(f"Model {model_id} not found")
                
                session_id = f"session_{model_id}_{int(time.time())}_{os.urandom(4).hex()}"
                
                session = InferenceSession(
                    session_id=session_id,
                    model_id=model_id,
                    encryption_context=security_level,
                    created_at=datetime.utcnow(),
                    active=True,
                    request_count=0,
                    total_inference_time=0.0
                )
                
                self.active_sessions[session_id] = session
                
                logger.info(f"Created inference session {session_id} for model {model_id}")
                return session_id
                
            except Exception as e:
                logger.error(f"Failed to create inference session: {e}")
                raise
    
    def run_encrypted_inference(
        self, 
        session_id: str, 
        encrypted_input: Union[ts.CKKSVector, List[float], np.ndarray]
    ) -> InferenceResult:
        """
        Run inference on encrypted data
        """
        with self._lock:
            try:
                # Validate session
                if session_id not in self.active_sessions:
                    raise ValueError(f"Session {session_id} not found or inactive")
                
                session = self.active_sessions[session_id]
                if not session.active:
                    raise ValueError(f"Session {session_id} is not active")
                
                # Get model
                model_id = session.model_id
                if model_id not in self.encrypted_models:
                    raise ValueError(f"Model {model_id} not found")
                
                encrypted_model = self.encrypted_models[model_id]
                secure_nn = encrypted_model.secure_nn
                
                start_time = time.time()
                
                # Ensure input is encrypted
                if isinstance(encrypted_input, (list, np.ndarray)):
                    logger.info("Encrypting input data...")
                    encryption_start = time.time()
                    
                    op_manager = self.op_managers[session.encryption_context]
                    if isinstance(encrypted_input, np.ndarray):
                        encrypted_input = encrypted_input.tolist()
                    
                    encrypted_vector = op_manager.encrypt_vector(
                        encrypted_input, session.encryption_context
                    )
                    
                    encryption_time = time.time() - encryption_start
                    self.inference_metrics['encryption_time'].append(encryption_time)
                else:
                    encrypted_vector = encrypted_input
                    encryption_time = 0.0
                
                # Run encrypted inference
                inference_start = time.time()
                logger.info("Running encrypted inference...")
                
                op_manager = self.op_managers[session.encryption_context]
                encrypted_output = secure_nn.forward_encrypted(encrypted_vector, op_manager)
                
                inference_time = time.time() - inference_start
                total_time = time.time() - start_time
                
                # Update session statistics
                session.request_count += 1
                session.total_inference_time += total_time
                
                # Record performance metrics
                self.inference_metrics['inference_time'].append(inference_time)
                self.inference_metrics['total_time'].append(total_time)
                
                # Create result
                result = InferenceResult(
                    session_id=session_id,
                    model_id=model_id,
                    encrypted_output=encrypted_output,
                    inference_time=inference_time,
                    total_time=total_time,
                    encryption_time=encryption_time,
                    timestamp=datetime.utcnow(),
                    success=True
                )
                
                logger.info(f"Encrypted inference completed in {total_time:.3f}s")
                return result
                
            except Exception as e:
                logger.error(f"Encrypted inference failed: {e}")
                
                # Create error result
                error_result = InferenceResult(
                    session_id=session_id,
                    model_id=session.model_id if 'session' in locals() else "unknown",
                    encrypted_output=None,
                    inference_time=0.0,
                    total_time=0.0,
                    encryption_time=0.0,
                    timestamp=datetime.utcnow(),
                    success=False,
                    error_message=str(e)
                )
                
                return error_result
    
    def decrypt_result(
        self, 
        inference_result: InferenceResult,
        session_id: str
    ) -> np.ndarray:
        """
        Decrypt inference result
        """
        try:
            if not inference_result.success or inference_result.encrypted_output is None:
                raise ValueError("Cannot decrypt failed inference result")
            
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            decryption_start = time.time()
            
            # Decrypt the output
            op_manager = self.op_managers[session.encryption_context]
            decrypted_output = op_manager.decrypt_vector(
                inference_result.encrypted_output, session.encryption_context
            )
            
            decryption_time = time.time() - decryption_start
            self.inference_metrics['decryption_time'].append(decryption_time)
            
            logger.info(f"Result decrypted in {decryption_time:.3f}s")
            return decrypted_output
            
        except Exception as e:
            logger.error(f"Result decryption failed: {e}")
            raise
    
    def _save_encrypted_model(self, encrypted_model: EncryptedModel):
        """Save encrypted model to disk"""
        try:
            model_file = f"{self.model_store_path}/encrypted/{encrypted_model.model_id}.pkl"
            metadata_file = f"{self.model_store_path}/metadata/{encrypted_model.model_id}.json"
            
            # Save metadata
            metadata_dict = asdict(encrypted_model.model_metadata)
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            # Save encrypted model (excluding the actual encrypted vectors for now)
            model_data = {
                'model_id': encrypted_model.model_id,
                'model_type': encrypted_model.model_type,
                'encryption_context': encrypted_model.encryption_context,
                'created_at': encrypted_model.created_at.isoformat(),
                'performance_metrics': encrypted_model.performance_metrics
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Encrypted model {encrypted_model.model_id} saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save encrypted model: {e}")
            raise
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get inference engine statistics"""
        with self._lock:
            active_sessions_count = len([s for s in self.active_sessions.values() if s.active])
            total_requests = sum(s.request_count for s in self.active_sessions.values())
            
            # Calculate performance statistics
            perf_stats = {}
            for metric, times in self.inference_metrics.items():
                if times:
                    perf_stats[metric] = {
                        'count': len(times),
                        'avg': np.mean(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'p95': np.percentile(times, 95)
                    }
            
            return {
                'loaded_models': len(self.encrypted_models),
                'active_sessions': active_sessions_count,
                'total_sessions': len(self.active_sessions),
                'total_requests': total_requests,
                'performance_metrics': perf_stats,
                'supported_security_levels': list(self.op_managers.keys())
            }
    
    def close_session(self, session_id: str):
        """Close an inference session"""
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].active = False
                logger.info(f"Session {session_id} closed")
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Clean up old inactive sessions"""
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            sessions_to_remove = []
            for session_id, session in self.active_sessions.items():
                if not session.active or session.created_at < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
            
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
