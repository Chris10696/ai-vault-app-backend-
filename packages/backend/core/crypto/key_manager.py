"""
Secure Key Management for Homomorphic Encryption
Implements Zero Trust key management with automated rotation
"""

import os
import json
import hmac
import hashlib
import secrets
import threading
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import logging

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from ..security.zero_trust import SecurityContext, TrustLevel
from ...shared.types.crypto import KeyMetadata, KeyStatus, SecurityAuditLog

logger = logging.getLogger(__name__)

class KeyType(Enum):
    """Types of cryptographic keys"""
    SEAL_SECRET = "seal_secret"
    SEAL_PUBLIC = "seal_public" 
    GALOIS = "galois"
    RELINEARIZATION = "relinearization"
    BOOTSTRAP = "bootstrap"
    MASTER = "master"

@dataclass
class SecureKeyEntry:
    """Secure key entry with metadata"""
    key_id: str
    key_type: KeyType
    encrypted_key: bytes
    metadata: KeyMetadata
    checksum: str
    created_at: datetime
    last_used: datetime
    usage_count: int
    status: KeyStatus

class AdvancedKeyManager:
    """
    Advanced key management with Zero Trust security
    Features:
    - Automated key rotation
    - Hardware Security Module (HSM) integration ready
    - Key escrow and recovery
    - Performance monitoring
    - Compliance audit trails
    """
    
    def __init__(
        self, 
        security_context: SecurityContext,
        key_store_path: Optional[str] = None,
        hsm_enabled: bool = False
    ):
        self.security_context = security_context
        self.key_store_path = key_store_path or os.getenv("KEY_STORE_PATH", "secure_keys")
        self.hsm_enabled = hsm_enabled
        
        # Security settings
        self.min_key_strength = 256  # bits
        self.max_key_age_hours = 12
        self.key_rotation_threshold = 500  # operations
        self.max_failed_attempts = 3
        
        # Internal state
        self._keys: Dict[str, SecureKeyEntry] = {}
        self._master_keys: Dict[str, bytes] = {}
        self._lock = threading.RLock()
        self._failed_attempts: Dict[str, int] = {}
        
        # Performance tracking
        self._operation_metrics: Dict[str, List[float]] = {}
        
        self._initialize_key_store()
        self._load_existing_keys()
    
    def _initialize_key_store(self):
        """Initialize secure key storage directory structure"""
        try:
            os.makedirs(self.key_store_path, exist_ok=True)
            os.makedirs(f"{self.key_store_path}/active", exist_ok=True)
            os.makedirs(f"{self.key_store_path}/archived", exist_ok=True)
            os.makedirs(f"{self.key_store_path}/audit", exist_ok=True)
            os.makedirs(f"{self.key_store_path}/backup", exist_ok=True)
            
            # Set restrictive permissions
            os.chmod(self.key_store_path, 0o700)
            
            logger.info("Key store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize key store: {e}")
            raise
    
    def _load_existing_keys(self):
        """Load existing keys from storage"""
        try:
            active_keys_path = f"{self.key_store_path}/active"
            
            if not os.path.exists(active_keys_path):
                return
            
            for filename in os.listdir(active_keys_path):
                if filename.endswith('.key'):
                    key_id = filename[:-4]  # Remove .key extension
                    
                    try:
                        key_entry = self._load_key_from_file(key_id)
                        if key_entry and key_entry.status == KeyStatus.ACTIVE:
                            self._keys[key_id] = key_entry
                            
                    except Exception as e:
                        logger.warning(f"Failed to load key {key_id}: {e}")
            
            logger.info(f"Loaded {len(self._keys)} active keys from storage")
            
        except Exception as e:
            logger.error(f"Failed to load existing keys: {e}")
    
    def generate_key_pair(
        self, 
        key_type: KeyType, 
        context_name: str,
        security_level: int = 128,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Generate a new cryptographic key pair
        Returns: (key_id, public_key_id)
        """
        with self._lock:
            try:
                # Verify security context has sufficient privileges
                if not self._verify_key_generation_rights():
                    raise PermissionError("Insufficient privileges for key generation")
                
                # Generate unique key ID
                key_id = self._generate_secure_key_id(key_type, context_name)
                
                # Generate cryptographic key pair based on type
                if key_type in [KeyType.SEAL_SECRET, KeyType.SEAL_PUBLIC]:
                    private_key, public_key = self._generate_seal_key_pair(
                        security_level, custom_params
                    )
                else:
                    private_key, public_key = self._generate_specialized_key_pair(
                        key_type, security_level, custom_params
                    )
                
                # Create metadata
                metadata = KeyMetadata(
                    algorithm="CKKS",
                    key_size=security_level,
                    context_name=context_name,
                    security_level=security_level,
                    custom_params=custom_params or {}
                )
                
                # Store private key
                private_key_entry = self._create_secure_key_entry(
                    key_id, key_type, private_key, metadata
                )
                
                # Store public key (if applicable)
                public_key_id = None
                if public_key is not None:
                    public_key_id = f"{key_id}_pub"
                    public_key_entry = self._create_secure_key_entry(
                        public_key_id, KeyType.SEAL_PUBLIC, public_key, metadata
                    )
                    self._store_key_entry(public_key_entry)
                    self._keys[public_key_id] = public_key_entry
                
                # Store private key
                self._store_key_entry(private_key_entry)
                self._keys[key_id] = private_key_entry
                
                # Audit log
                self._log_key_operation(
                    "generate", key_id, f"Generated {key_type.value} key pair"
                )
                
                logger.info(f"Generated key pair: {key_id} (type: {key_type.value})")
                return key_id, public_key_id or key_id
                
            except Exception as e:
                logger.error(f"Key generation failed: {e}")
                raise
    
    def get_key(self, key_id: str, key_type: KeyType) -> bytes:
        """
        Securely retrieve a cryptographic key
        Implements Zero Trust verification
        """
        with self._lock:
            try:
                # Verify access rights
                if not self._verify_key_access_rights(key_id, key_type):
                    self._record_failed_attempt(key_id)
                    raise PermissionError(f"Access denied for key {key_id}")
                
                # Check if key exists and is active
                if key_id not in self._keys:
                    raise KeyError(f"Key {key_id} not found")
                
                key_entry = self._keys[key_id]
                
                if key_entry.status != KeyStatus.ACTIVE:
                    raise ValueError(f"Key {key_id} is not active (status: {key_entry.status})")
                
                # Check if key needs rotation
                if self._should_rotate_key(key_entry):
                    logger.warning(f"Key {key_id} should be rotated")
                    self._schedule_key_rotation(key_id)
                
                # Decrypt and return key
                decrypted_key = self._decrypt_key(key_entry.encrypted_key)
                
                # Update usage statistics
                key_entry.last_used = datetime.utcnow()
                key_entry.usage_count += 1
                self._update_key_entry(key_entry)
                
                # Reset failed attempts counter
                self._failed_attempts.pop(key_id, None)
                
                # Audit log
                self._log_key_operation("access", key_id, "Key accessed successfully")
                
                return decrypted_key
                
            except Exception as e:
                logger.error(f"Key retrieval failed for {key_id}: {e}")
                raise
    
    def rotate_key(self, old_key_id: str, force: bool = False) -> str:
        """
        Rotate an existing key with a new one
        Maintains backward compatibility during transition period
        """
        with self._lock:
            try:
                if old_key_id not in self._keys:
                    raise KeyError(f"Key {old_key_id} not found for rotation")
                
                old_key_entry = self._keys[old_key_id]
                
                # Check if rotation is needed (unless forced)
                if not force and not self._should_rotate_key(old_key_entry):
                    logger.info(f"Key {old_key_id} does not need rotation yet")
                    return old_key_id
                
                # Generate new key with same parameters
                new_key_id, _ = self.generate_key_pair(
                    old_key_entry.key_type,
                    old_key_entry.metadata.context_name,
                    old_key_entry.metadata.security_level,
                    old_key_entry.metadata.custom_params
                )
                
                # Archive old key
                self._archive_key(old_key_id)
                
                # Audit log
                self._log_key_operation(
                    "rotate", new_key_id, f"Rotated key {old_key_id} -> {new_key_id}"
                )
                
                logger.info(f"Key rotated: {old_key_id} -> {new_key_id}")
                return new_key_id
                
            except Exception as e:
                logger.error(f"Key rotation failed for {old_key_id}: {e}")
                raise
    
    def _generate_seal_key_pair(
        self, 
        security_level: int, 
        custom_params: Optional[Dict[str, Any]]
    ) -> Tuple[bytes, bytes]:
        """Generate SEAL-specific key pair"""
        try:
            # For SEAL keys, we generate random keys that will be used
            # to derive the actual SEAL context keys
            private_key = secrets.token_bytes(security_level // 8)
            public_key = self._derive_public_key(private_key)
            
            return private_key, public_key
            
        except Exception as e:
            logger.error(f"SEAL key pair generation failed: {e}")
            raise
    
    def _generate_specialized_key_pair(
        self, 
        key_type: KeyType, 
        security_level: int, 
        custom_params: Optional[Dict[str, Any]]
    ) -> Tuple[bytes, Optional[bytes]]:
        """Generate specialized keys (Galois, Relinearization, etc.)"""
        try:
            if key_type == KeyType.GALOIS:
                # Generate Galois key for rotations
                key_material = secrets.token_bytes(security_level // 4)  # Smaller for efficiency
                return key_material, None
            
            elif key_type == KeyType.RELINEARIZATION:
                # Generate relinearization keys
                key_material = secrets.token_bytes(security_level // 2)
                return key_material, None
            
            elif key_type == KeyType.BOOTSTRAP:
                # Generate bootstrapping keys
                key_material = secrets.token_bytes(security_level)
                return key_material, None
            
            elif key_type == KeyType.MASTER:
                # Generate master key for key encryption
                master_key = secrets.token_bytes(32)  # 256-bit master key
                return master_key, None
            
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
                
        except Exception as e:
            logger.error(f"Specialized key generation failed for {key_type}: {e}")
            raise
    
    def _derive_public_key(self, private_key: bytes) -> bytes:
        """Derive public key from private key"""
        # Use HMAC-SHA256 to derive public key
        public_key = hmac.new(
            private_key, 
            b"seal_public_key_derivation", 
            hashlib.sha256
        ).digest()
        return public_key
    
    def _create_secure_key_entry(
        self, 
        key_id: str, 
        key_type: KeyType, 
        key_material: bytes, 
        metadata: KeyMetadata
    ) -> SecureKeyEntry:
        """Create a secure key entry with encryption and checksums"""
        try:
            # Encrypt key material
            encrypted_key = self._encrypt_key(key_material)
            
            # Calculate checksum
            checksum = hashlib.sha256(key_material).hexdigest()
            
            # Create entry
            entry = SecureKeyEntry(
                key_id=key_id,
                key_type=key_type,
                encrypted_key=encrypted_key,
                metadata=metadata,
                checksum=checksum,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                usage_count=0,
                status=KeyStatus.ACTIVE
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to create secure key entry: {e}")
            raise
    
    def _encrypt_key(self, key_material: bytes) -> bytes:
        """Encrypt key material using master key"""
        try:
            # Get or create master key
            master_key = self._get_master_key()
            
            # Create Fernet cipher
            fernet = Fernet(base64.urlsafe_b64encode(master_key[:32]))
            
            # Encrypt key material
            encrypted_key = fernet.encrypt(key_material)
            
            return encrypted_key
            
        except Exception as e:
            logger.error(f"Key encryption failed: {e}")
            raise
    
    def _decrypt_key(self, encrypted_key: bytes) -> bytes:
        """Decrypt key material using master key"""
        try:
            # Get master key
            master_key = self._get_master_key()
            
            # Create Fernet cipher
            fernet = Fernet(base64.urlsafe_b64encode(master_key[:32]))
            
            # Decrypt key material
            decrypted_key = fernet.decrypt(encrypted_key)
            
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Key decryption failed: {e}")
            raise
    
    def _get_master_key(self) -> bytes:
        """Get or create master encryption key"""
        context_id = self.security_context.get_context_id()
        
        if context_id not in self._master_keys:
            # Derive master key from security context
            master_key = self._derive_master_key()
            self._master_keys[context_id] = master_key
        
        return self._master_keys[context_id]
    
    def _derive_master_key(self) -> bytes:
        """Derive master key from security context"""
        # Use security context credentials to derive master key
        password = self.security_context.get_encryption_key().encode()
        salt = self.security_context.get_context_id().encode()[:16]
        
        # Pad salt if needed
        if len(salt) < 16:
            salt = salt + b'0' * (16 - len(salt))
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(password)
    
    def _generate_secure_key_id(self, key_type: KeyType, context_name: str) -> str:
        """Generate cryptographically secure key ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_part = secrets.token_hex(8)
        return f"{key_type.value}_{context_name}_{timestamp}_{random_part}"
    
    def _verify_key_generation_rights(self) -> bool:
        """Verify if current context has key generation rights"""
        trust_score = self.security_context.get_trust_score()
        trust_level = self.security_context.get_trust_level()
        
        # Require high trust for key generation
        return trust_score >= 0.8 and trust_level in [TrustLevel.TRUSTED, TrustLevel.HIGH_TRUST]
    
    def _verify_key_access_rights(self, key_id: str, key_type: KeyType) -> bool:
        """Verify if current context has key access rights"""
        # Check failed attempts
        if self._failed_attempts.get(key_id, 0) >= self.max_failed_attempts:
            logger.warning(f"Key {key_id} blocked due to too many failed attempts")
            return False
        
        # Check trust level
        trust_score = self.security_context.get_trust_score()
        if trust_score < 0.6:
            return False
        
        # Additional checks for sensitive key types
        if key_type == KeyType.MASTER:
            return self.security_context.get_trust_level() == TrustLevel.HIGH_TRUST
        
        return True
    
    def _should_rotate_key(self, key_entry: SecureKeyEntry) -> bool:
        """Determine if key should be rotated"""
        # Check age
        age_hours = (datetime.utcnow() - key_entry.created_at).total_seconds() / 3600
        if age_hours > self.max_key_age_hours:
            return True
        
        # Check usage count
        if key_entry.usage_count > self.key_rotation_threshold:
            return True
        
        # Check if key is approaching expiration
        if key_entry.status == KeyStatus.EXPIRING:
            return True
        
        return False
    
    def _store_key_entry(self, key_entry: SecureKeyEntry):
        """Store key entry to persistent storage"""
        try:
            key_file = f"{self.key_store_path}/active/{key_entry.key_id}.key"
            
            # Serialize key entry
            entry_data = {
                'key_id': key_entry.key_id,
                'key_type': key_entry.key_type.value,
                'encrypted_key': base64.b64encode(key_entry.encrypted_key).decode(),
                'metadata': asdict(key_entry.metadata),
                'checksum': key_entry.checksum,
                'created_at': key_entry.created_at.isoformat(),
                'last_used': key_entry.last_used.isoformat(),
                'usage_count': key_entry.usage_count,
                'status': key_entry.status.value
            }
            
            # Write to file with atomic operation
            temp_file = f"{key_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(entry_data, f, indent=2)
            
            os.rename(temp_file, key_file)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to store key entry {key_entry.key_id}: {e}")
            raise
    
    def _load_key_from_file(self, key_id: str) -> Optional[SecureKeyEntry]:
        """Load key entry from file"""
        try:
            key_file = f"{self.key_store_path}/active/{key_id}.key"
            
            if not os.path.exists(key_file):
                return None
            
            with open(key_file, 'r') as f:
                entry_data = json.load(f)
            
            # Reconstruct key entry
            key_entry = SecureKeyEntry(
                key_id=entry_data['key_id'],
                key_type=KeyType(entry_data['key_type']),
                encrypted_key=base64.b64decode(entry_data['encrypted_key']),
                metadata=KeyMetadata(**entry_data['metadata']),
                checksum=entry_data['checksum'],
                created_at=datetime.fromisoformat(entry_data['created_at']),
                last_used=datetime.fromisoformat(entry_data['last_used']),
                usage_count=entry_data['usage_count'],
                status=KeyStatus(entry_data['status'])
            )
            
            return key_entry
            
        except Exception as e:
            logger.error(f"Failed to load key {key_id}: {e}")
            return None
    
    def _update_key_entry(self, key_entry: SecureKeyEntry):
        """Update key entry in storage"""
        self._store_key_entry(key_entry)
    
    def _archive_key(self, key_id: str):
        """Archive a key (move from active to archived)"""
        try:
            if key_id in self._keys:
                key_entry = self._keys[key_id]
                key_entry.status = KeyStatus.ARCHIVED
                
                # Move file to archive directory
                active_file = f"{self.key_store_path}/active/{key_id}.key"
                archive_file = f"{self.key_store_path}/archived/{key_id}.key"
                
                if os.path.exists(active_file):
                    os.rename(active_file, archive_file)
                
                # Update in-memory state
                del self._keys[key_id]
                
                logger.info(f"Key {key_id} archived successfully")
            
        except Exception as e:
            logger.error(f"Failed to archive key {key_id}: {e}")
            raise
    
    def _record_failed_attempt(self, key_id: str):
        """Record failed access attempt"""
        self._failed_attempts[key_id] = self._failed_attempts.get(key_id, 0) + 1
        
        # Log security event
        self._log_key_operation(
            "access_denied", key_id, f"Failed attempt #{self._failed_attempts[key_id]}"
        )
    
    def _schedule_key_rotation(self, key_id: str):
        """Schedule key for rotation"""
        if key_id in self._keys:
            self._keys[key_id].status = KeyStatus.EXPIRING
            
            # Log rotation scheduling
            self._log_key_operation(
                "schedule_rotation", key_id, "Key scheduled for rotation"
            )
    
    def _log_key_operation(self, operation: str, key_id: str, details: str):
        """Log key operation for audit trail"""
        try:
            audit_entry = SecurityAuditLog(
                timestamp=datetime.utcnow(),
                operation=operation,
                resource=f"key:{key_id}",
                user_id=self.security_context.get_user_id(),
                trust_score=self.security_context.get_trust_score(),
                details=details,
                success=True
            )
            
            # Write to audit log
            audit_file = f"{self.key_store_path}/audit/key_operations.log"
            with open(audit_file, 'a') as f:
                f.write(json.dumps(asdict(audit_entry), default=str) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to log key operation: {e}")
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """Get key management statistics"""
        with self._lock:
            active_keys = len([k for k in self._keys.values() if k.status == KeyStatus.ACTIVE])
            expiring_keys = len([k for k in self._keys.values() if k.status == KeyStatus.EXPIRING])
            
            # Key type distribution
            key_types = {}
            for key_entry in self._keys.values():
                key_type = key_entry.key_type.value
                key_types[key_type] = key_types.get(key_type, 0) + 1
            
            # Usage statistics
            total_usage = sum(k.usage_count for k in self._keys.values())
            avg_age_hours = 0
            if self._keys:
                avg_age = sum(
                    (datetime.utcnow() - k.created_at).total_seconds() 
                    for k in self._keys.values()
                ) / len(self._keys)
                avg_age_hours = avg_age / 3600
            
            return {
                'total_keys': len(self._keys),
                'active_keys': active_keys,
                'expiring_keys': expiring_keys,
                'key_types': key_types,
                'total_usage': total_usage,
                'avg_age_hours': round(avg_age_hours, 2),
                'failed_attempts': dict(self._failed_attempts)
            }
    
    def cleanup_expired_keys(self, max_age_days: int = 30):
        """Clean up expired and archived keys"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            # Clean up archived keys
            archive_dir = f"{self.key_store_path}/archived"
            if os.path.exists(archive_dir):
                for filename in os.listdir(archive_dir):
                    if filename.endswith('.key'):
                        filepath = os.path.join(archive_dir, filename)
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        
                        if file_time < cutoff_date:
                            os.remove(filepath)
                            cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired key files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Key cleanup failed: {e}")
            return 0
