"""
Shared type definitions for cryptographic operations
Used across backend and potentially frontend for type safety
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import numpy as np

try:
    import tenseal as ts
except ImportError:
    # Handle cases where tenseal is not available (e.g., frontend)
    ts = None

class EncryptionLevel(Enum):
    """Encryption security levels"""
    HIGH = "high_security"
    MEDIUM = "medium_security"
    PERFORMANCE = "performance"

class KeyStatus(Enum):
    """Key lifecycle status"""
    ACTIVE = "ACTIVE"
    EXPIRING = "EXPIRING"
    ARCHIVED = "ARCHIVED"
    REVOKED = "REVOKED"

class HomomorphicOperation(Enum):
    """Supported homomorphic operations"""
    ADD = "add"
    MULTIPLY = "multiply"
    MATRIX_MULTIPLY = "matrix_multiply"
    POLYNOMIAL = "polynomial"
    AVERAGE = "average"

@dataclass
class SealContextConfig:
    """Configuration for SEAL homomorphic encryption context"""
    scheme: str
    poly_modulus_degree: int
    coeff_modulus: List[int]
    scale_bits: int
    security_level: int

@dataclass
class SealKeyPair:
    """SEAL key pair with metadata"""
    context_name: str
    secret_key: Any  # ts.SecretKey when available
    public_key: Any  # ts.PublicKey when available
    created_at: datetime
    key_id: str
    security_level: int
    usage_count: int

@dataclass
class KeyMetadata:
    """Metadata for cryptographic keys"""
    algorithm: str
    key_size: int
    context_name: str
    security_level: int
    custom_params: Dict[str, Any]

@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    timestamp: datetime
    operation: str
    resource: str
    user_id: str
    trust_score: float
    details: str
    success: bool

@dataclass
class ModelMetadata:
    """Metadata for encrypted AI models"""
    model_id: str
    model_type: str
    input_shape: List[int]
    output_shape: List[int]
    security_level: str
    created_at: datetime
    performance_requirements: Dict[str, Any]

@dataclass
class InferenceRequest:
    """Request for homomorphic inference"""
    model_id: str
    input_data: Union[List[float], np.ndarray]
    security_level: str = "high_security"
    return_encrypted: bool = False
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class InferenceResult:
    """Result from homomorphic inference"""
    session_id: str
    model_id: str
    encrypted_output: Optional[Any]  # ts.CKKSVector when available
    inference_time: float
    total_time: float
    encryption_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    decrypted_output: Optional[np.ndarray] = None

@dataclass
class ModelDeployment:
    """Model deployment configuration"""
    model_id: str
    deployment_id: str
    security_level: str
    max_concurrent_sessions: int
    auto_scaling: bool
    performance_targets: Dict[str, float]
    created_at: datetime

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    name: str
    description: str
    min_trust_level: Optional[Any] = None  # TrustLevel when available
    rate_limit_per_hour: Optional[int] = None
    allowed_ips: Optional[List[str]] = None
    require_mfa: bool = False
    audit_required: bool = False

class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Type aliases for better code readability
EncryptedVector = Any  # ts.CKKSVector when available
EncryptedMatrix = List[Any]  # List[ts.CKKSVector] when available
FloatArray = Union[List[float], np.ndarray]
ModelWeights = Dict[str, np.ndarray]

# Constants
DEFAULT_SCALE_BITS = 40
DEFAULT_SECURITY_LEVEL = 128
MAX_BATCH_SIZE = 32
DEFAULT_TTL_HOURS = 24

# Validation functions
def validate_security_level(level: str) -> bool:
    """Validate security level string"""
    return level in [e.value for e in EncryptionLevel]

def validate_model_id(model_id: str) -> bool:
    """Validate model ID format"""
    if not model_id or len(model_id) < 3:
        return False
    return model_id.replace('_', '').replace('-', '').isalnum()

def validate_input_shape(shape: List[int]) -> bool:
    """Validate model input shape"""
    if not shape:
        return False
    return all(isinstance(dim, int) and dim > 0 for dim in shape)

# Helper functions
def create_default_context_config(security_level: str = "high_security") -> SealContextConfig:
    """Create default SEAL context configuration"""
    if security_level == "high_security":
        return SealContextConfig(
            scheme="CKKS",
            poly_modulus_degree=16384,
            coeff_modulus=[60, 40, 40, 40, 40, 40, 40, 60],
            scale_bits=DEFAULT_SCALE_BITS,
            security_level=128
        )
    elif security_level == "medium_security":
        return SealContextConfig(
            scheme="CKKS",
            poly_modulus_degree=8192,
            coeff_modulus=[60, 40, 40, 40, 40, 60],
            scale_bits=DEFAULT_SCALE_BITS,
            security_level=112
        )
    else:  # performance
        return SealContextConfig(
            scheme="CKKS",
            poly_modulus_degree=4096,
            coeff_modulus=[60, 40, 40, 60],
            scale_bits=DEFAULT_SCALE_BITS,
            security_level=100
        )

def estimate_computation_time(operation: HomomorphicOperation, input_size: int) -> float:
    """Estimate computation time for homomorphic operation"""
    base_times = {
        HomomorphicOperation.ADD: 0.001,
        HomomorphicOperation.MULTIPLY: 0.01,
        HomomorphicOperation.MATRIX_MULTIPLY: 0.1,
        HomomorphicOperation.POLYNOMIAL: 0.05,
        HomomorphicOperation.AVERAGE: 0.005
    }
    
    base_time = base_times.get(operation, 0.01)
    size_factor = np.log2(max(1, input_size)) / 10.0
    
    return base_time * (1 + size_factor)

def calculate_memory_requirements(
    operation: HomomorphicOperation, 
    input_size: int, 
    security_level: str = "high_security"
) -> int:
    """Calculate memory requirements in MB for operation"""
    base_memory = {
        "high_security": 100,
        "medium_security": 50,
        "performance": 25
    }
    
    base = base_memory.get(security_level, 50)
    size_factor = input_size / 1000.0  # Scale by input size
    
    operation_multipliers = {
        HomomorphicOperation.ADD: 1.0,
        HomomorphicOperation.MULTIPLY: 1.5,
        HomomorphicOperation.MATRIX_MULTIPLY: 3.0,
        HomomorphicOperation.POLYNOMIAL: 2.0,
        HomomorphicOperation.AVERAGE: 1.2
    }
    
    multiplier = operation_multipliers.get(operation, 1.0)
    
    return int(base * multiplier * (1 + size_factor))
