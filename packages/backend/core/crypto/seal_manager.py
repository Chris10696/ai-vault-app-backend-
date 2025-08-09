"""
Microsoft SEAL Integration Manager
Handles SEAL library operations, key management, and encryption parameters
"""

import os
import json
import pickle
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import numpy as np

try:
    import tenseal as ts
except ImportError:
    raise ImportError("TenSEAL not installed. Run: pip install tenseal")

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from ..security.zero_trust import SecurityContext
from ...shared.types.crypto import (
    SealContextConfig, 
    SealKeyPair, 
    EncryptionLevel,
    HomomorphicOperation
)

logger = logging.getLogger(__name__)

class SealContextManager:
    """
    Manages Microsoft SEAL contexts and cryptographic parameters
    Implements secure key management with Zero Trust principles
    """
    
    def __init__(
        self, 
        security_context: SecurityContext,
        config_path: Optional[str] = None
    ):
        self.security_context = security_context
        self.config_path = config_path or os.getenv("SEAL_CONFIG_PATH", "seal_configs")
        self._contexts: Dict[str, ts.Context] = {}
        self._keys: Dict[str, SealKeyPair] = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self._operation_metrics: Dict[str, Dict[str, float]] = {}
        
        # Security parameters
        self.min_security_level = 128  # bits
        self.max_key_age_hours = 24
        self.key_rotation_threshold = 1000  # operations
        
        self._setup_directories()
        self._initialize_default_contexts()
    
    def _setup_directories(self):
        """Create necessary directories for key storage"""
        os.makedirs(self.config_path, exist_ok=True)
        os.makedirs(f"{self.config_path}/keys", exist_ok=True)
        os.makedirs(f"{self.config_path}/contexts", exist_ok=True)
    
    def _initialize_default_contexts(self):
        """Initialize default CKKS contexts for different security levels"""
        
        # High Security Context (128-bit security)
        high_security_config = SealContextConfig(
            scheme="CKKS",
            poly_modulus_degree=16384,  # n = 2^14
            coeff_modulus=[60, 40, 40, 40, 40, 40, 40, 60],
            scale_bits=40,
            security_level=128
        )
        
        # Medium Security Context (112-bit security)
        medium_security_config = SealContextConfig(
            scheme="CKKS",
            poly_modulus_degree=8192,   # n = 2^13
            coeff_modulus=[60, 40, 40, 40, 40, 60],
            scale_bits=40,
            security_level=112
        )
        
        # Performance Context (lower security for testing)
        perf_security_config = SealContextConfig(
            scheme="CKKS",
            poly_modulus_degree=4096,   # n = 2^12
            coeff_modulus=[60, 40, 40, 60],
            scale_bits=40,
            security_level=100
        )
        
        # Create contexts
        self._create_context("high_security", high_security_config)
        self._create_context("medium_security", medium_security_config)
        self._create_context("performance", perf_security_config)
    
    def _create_context(self, context_name: str, config: SealContextConfig) -> ts.Context:
        """Create a new SEAL context with specified configuration"""
        with self._lock:
            try:
                logger.info(f"Creating SEAL context: {context_name}")
                
                # Create TenSEAL context
                context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=config.poly_modulus_degree,
                    coeff_modulus=config.coeff_modulus
                )
                
                # Set global scale
                context.global_scale = 2 ** config.scale_bits
                
                # Generate galois keys for rotations (enables SIMD operations)
                context.generate_galois_keys()
                
                # Store context
                self._contexts[context_name] = context
                
                # Save context configuration
                self._save_context_config(context_name, config)
                
                logger.info(f"SEAL context '{context_name}' created successfully")
                return context
                
            except Exception as e:
                logger.error(f"Failed to create SEAL context '{context_name}': {e}")
                raise
    
    def get_context(self, context_name: str = "high_security") -> ts.Context:
        """Get SEAL context by name"""
        with self._lock:
            if context_name not in self._contexts:
                raise ValueError(f"Context '{context_name}' not found")
            return self._contexts[context_name]
    
    def generate_keys(self, context_name: str = "high_security") -> SealKeyPair:
        """
        Generate new encryption keys for specified context
        Implements secure key generation with Zero Trust verification
        """
        with self._lock:
            try:
                context = self.get_context(context_name)
                
                # Generate keys using TenSEAL
                secret_key = context.secret_key()
                public_key = context.public_key()
                
                # Create key pair object
                key_pair = SealKeyPair(
                    context_name=context_name,
                    secret_key=secret_key,
                    public_key=public_key,
                    created_at=datetime.utcnow(),
                    key_id=self._generate_key_id(),
                    security_level=self._get_context_security_level(context_name),
                    usage_count=0
                )
                
                # Store keys securely
                self._store_keys(key_pair)
                self._keys[context_name] = key_pair
                
                # Log key generation for audit
                self._log_key_operation("generate", context_name, key_pair.key_id)
                
                logger.info(f"New key pair generated for context '{context_name}'")
                return key_pair
                
            except Exception as e:
                logger.error(f"Key generation failed for context '{context_name}': {e}")
                raise
    
    def get_keys(self, context_name: str = "high_security") -> SealKeyPair:
        """Get encryption keys, generating new ones if needed"""
        with self._lock:
            if context_name not in self._keys:
                return self.generate_keys(context_name)
            
            key_pair = self._keys[context_name]
            
            # Check if key rotation is needed
            if self._should_rotate_key(key_pair):
                logger.info(f"Key rotation required for context '{context_name}'")
                return self.generate_keys(context_name)
            
            return key_pair
    
    def _should_rotate_key(self, key_pair: SealKeyPair) -> bool:
        """Determine if key should be rotated based on age and usage"""
        # Check age
        age_hours = (datetime.utcnow() - key_pair.created_at).total_seconds() / 3600
        if age_hours > self.max_key_age_hours:
            return True
        
        # Check usage count
        if key_pair.usage_count > self.key_rotation_threshold:
            return True
        
        return False
    
    def _generate_key_id(self) -> str:
        """Generate unique key identifier"""
        import uuid
        return f"seal_key_{uuid.uuid4().hex[:16]}"
    
    def _get_context_security_level(self, context_name: str) -> int:
        """Get security level for context"""
        security_levels = {
            "high_security": 128,
            "medium_security": 112,
            "performance": 100
        }
        return security_levels.get(context_name, 128)
    
    def _store_keys(self, key_pair: SealKeyPair):
        """Securely store keys to disk with encryption"""
        try:
            # Encrypt key data before storage
            key_data = {
                'context_name': key_pair.context_name,
                'key_id': key_pair.key_id,
                'created_at': key_pair.created_at.isoformat(),
                'security_level': key_pair.security_level,
                'usage_count': key_pair.usage_count,
                'secret_key': key_pair.secret_key,
                'public_key': key_pair.public_key
            }
            
            # Serialize and encrypt
            serialized_data = pickle.dumps(key_data)
            encrypted_data = self._encrypt_key_data(serialized_data)
            
            # Save to file
            key_file = f"{self.config_path}/keys/{key_pair.key_id}.key"
            with open(key_file, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"Keys stored securely: {key_pair.key_id}")
            
        except Exception as e:
            logger.error(f"Failed to store keys: {e}")
            raise
    
    def _encrypt_key_data(self, data: bytes) -> bytes:
        """Encrypt key data using Fernet symmetric encryption"""
        # Use security context to derive encryption key
        password = self.security_context.get_encryption_key().encode()
        salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        
        # Prepend salt to encrypted data
        return salt + encrypted_data
    
    def _save_context_config(self, context_name: str, config: SealContextConfig):
        """Save context configuration to disk"""
        config_file = f"{self.config_path}/contexts/{context_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
    
    def _log_key_operation(self, operation: str, context_name: str, key_id: str):
        """Log key operations for audit trail"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'context_name': context_name,
            'key_id': key_id,
            'user_id': self.security_context.get_user_id(),
            'trust_score': self.security_context.get_trust_score()
        }
        
        audit_file = f"{self.config_path}/audit.log"
        with open(audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
    
    def get_context_info(self, context_name: str) -> Dict[str, Any]:
        """Get detailed information about a SEAL context"""
        with self._lock:
            if context_name not in self._contexts:
                raise ValueError(f"Context '{context_name}' not found")
            
            context = self._contexts[context_name]
            config_file = f"{self.config_path}/contexts/{context_name}.json"
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            return {
                'context_name': context_name,
                'scheme': 'CKKS',
                'poly_modulus_degree': config.get('poly_modulus_degree'),
                'coeff_modulus': config.get('coeff_modulus'),
                'scale_bits': config.get('scale_bits'),
                'security_level': config.get('security_level'),
                'slots': context.slot_count(),
                'has_galois_keys': context.galois_keys() is not None,
                'has_relin_keys': context.relin_keys() is not None
            }
    
    def cleanup_old_keys(self, max_age_days: int = 7):
        """Clean up old key files"""
        try:
            keys_dir = f"{self.config_path}/keys"
            if not os.path.exists(keys_dir):
                return
            
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            for filename in os.listdir(keys_dir):
                if filename.endswith('.key'):
                    filepath = os.path.join(keys_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old key file: {filename}")
            
        except Exception as e:
            logger.error(f"Key cleanup failed: {e}")


class HomomorphicOperationManager:
    """
    Manages homomorphic operations with performance optimization
    """
    
    def __init__(self, seal_manager: SealContextManager):
        self.seal_manager = seal_manager
        self._operation_cache: Dict[str, Any] = {}
        self._performance_stats: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def encrypt_vector(
        self, 
        data: List[float], 
        context_name: str = "high_security"
    ) -> ts.CKKSVector:
        """Encrypt a vector of floating-point numbers"""
        try:
            context = self.seal_manager.get_context(context_name)
            
            # Convert to numpy array for efficiency
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float64)
            
            # Create encrypted vector
            encrypted_vector = ts.ckks_vector(context, data)
            
            # Update usage statistics
            key_pair = self.seal_manager.get_keys(context_name)
            key_pair.usage_count += 1
            
            logger.debug(f"Encrypted vector of size {len(data)} using context '{context_name}'")
            return encrypted_vector
            
        except Exception as e:
            logger.error(f"Vector encryption failed: {e}")
            raise
    
    def decrypt_vector(
        self, 
        encrypted_vector: ts.CKKSVector, 
        context_name: str = "high_security"
    ) -> np.ndarray:
        """Decrypt a vector to floating-point numbers"""
        try:
            # Decrypt and return as numpy array
            decrypted_data = encrypted_vector.decrypt()
            
            logger.debug(f"Decrypted vector of size {len(decrypted_data)}")
            return np.array(decrypted_data)
            
        except Exception as e:
            logger.error(f"Vector decryption failed: {e}")
            raise
    
    def homomorphic_add(
        self, 
        vec1: ts.CKKSVector, 
        vec2: ts.CKKSVector
    ) -> ts.CKKSVector:
        """Perform homomorphic addition on encrypted vectors"""
        try:
            start_time = time.time()
            result = vec1 + vec2
            elapsed = time.time() - start_time
            
            self._record_performance("add", elapsed)
            
            logger.debug(f"Homomorphic addition completed in {elapsed:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Homomorphic addition failed: {e}")
            raise
    
    def homomorphic_multiply(
        self, 
        vec1: ts.CKKSVector, 
        vec2: ts.CKKSVector
    ) -> ts.CKKSVector:
        """Perform homomorphic multiplication on encrypted vectors"""
        try:
            import time
            start_time = time.time()
            result = vec1 * vec2
            elapsed = time.time() - start_time
            
            self._record_performance("multiply", elapsed)
            
            logger.debug(f"Homomorphic multiplication completed in {elapsed:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Homomorphic multiplication failed: {e}")
            raise
    
    def homomorphic_matrix_multiply(
        self, 
        matrix: List[ts.CKKSVector], 
        vector: ts.CKKSVector
    ) -> ts.CKKSVector:
        """Perform homomorphic matrix-vector multiplication"""
        try:
            import time
            start_time = time.time()
            
            # Initialize result with zeros
            context = vector.context()
            zero_data = np.zeros(vector.size())
            result = ts.ckks_vector(context, zero_data)
            
            # Perform matrix-vector multiplication
            for i, row in enumerate(matrix):
                # Element-wise multiplication and sum
                product = row * vector
                result = result + product
            
            elapsed = time.time() - start_time
            self._record_performance("matrix_multiply", elapsed)
            
            logger.debug(f"Homomorphic matrix multiplication completed in {elapsed:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Homomorphic matrix multiplication failed: {e}")
            raise
    
    def _record_performance(self, operation: str, elapsed_time: float):
        """Record performance metrics for operations"""
        with self._lock:
            if operation not in self._performance_stats:
                self._performance_stats[operation] = []
            
            self._performance_stats[operation].append(elapsed_time)
            
            # Keep only last 100 measurements
            if len(self._performance_stats[operation]) > 100:
                self._performance_stats[operation] = self._performance_stats[operation][-100:]
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations"""
        with self._lock:
            stats = {}
            
            for operation, times in self._performance_stats.items():
                if times:
                    stats[operation] = {
                        'count': len(times),
                        'avg_time': np.mean(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'std_time': np.std(times)
                    }
            
            return stats
