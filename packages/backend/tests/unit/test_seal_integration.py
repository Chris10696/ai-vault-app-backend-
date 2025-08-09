"""
Unit tests for SEAL integration and key management
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from packages.backend.core.crypto.seal_manager import (
    SealContextManager, 
    HomomorphicOperationManager
)
from packages.backend.core.crypto.key_manager import (
    AdvancedKeyManager,
    KeyType,
    SecureKeyEntry
)
from packages.backend.core.security.zero_trust import SecurityContext, TrustLevel
from packages.backend.shared.types.crypto import SealContextConfig, KeyMetadata, KeyStatus

class TestSealIntegration:
    """Test SEAL library integration"""
    
    @pytest.fixture
    def mock_security_context(self):
        """Mock security context for testing"""
        context = Mock(spec=SecurityContext)
        context.get_user_id.return_value = "test_user"
        context.get_trust_score.return_value = 0.9
        context.get_trust_level.return_value = TrustLevel.TRUSTED
        context.get_context_id.return_value = "test_context"
        context.get_encryption_key.return_value = "test_encryption_key"
        return context
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def seal_manager(self, mock_security_context, temp_config_dir):
        """Create SEAL manager instance for testing"""
        return SealContextManager(mock_security_context, temp_config_dir)
    
    def test_context_creation(self, seal_manager):
        """Test SEAL context creation"""
        # Test getting default context
        context = seal_manager.get_context("high_security")
        assert context is not None
        
        # Test context properties
        info = seal_manager.get_context_info("high_security")
        assert info['scheme'] == 'CKKS'
        assert info['poly_modulus_degree'] == 16384
        assert info['security_level'] == 128
        assert info['slots'] > 0
    
    def test_key_generation(self, seal_manager):
        """Test key pair generation"""
        # Generate keys
        key_pair = seal_manager.generate_keys("high_security")
        
        assert key_pair is not None
        assert key_pair.context_name == "high_security"
        assert key_pair.security_level == 128
        assert key_pair.secret_key is not None
        assert key_pair.public_key is not None
        assert isinstance(key_pair.created_at, datetime)
        assert key_pair.usage_count == 0
    
    def test_homomorphic_operations(self, seal_manager):
        """Test basic homomorphic operations"""
        # Create operation manager
        op_manager = HomomorphicOperationManager(seal_manager)
        
        # Test data
        data1 = [1.5, 2.5, 3.5, 4.5]
        data2 = [0.5, 1.0, 1.5, 2.0]
        
        # Encrypt vectors
        vec1 = op_manager.encrypt_vector(data1, "performance")  # Use faster context for testing
        vec2 = op_manager.encrypt_vector(data2, "performance")
        
        assert vec1 is not None
        assert vec2 is not None
        
        # Test homomorphic addition
        result_add = op_manager.homomorphic_add(vec1, vec2)
        decrypted_add = op_manager.decrypt_vector(result_add, "performance")
        
        expected_add = np.array(data1) + np.array(data2)
        np.testing.assert_allclose(decrypted_add[:len(expected_add)], expected_add, rtol=1e-2)
        
        # Test homomorphic multiplication
        result_mul = op_manager.homomorphic_multiply(vec1, vec2)
        decrypted_mul = op_manager.decrypt_vector(result_mul, "performance")
        
        expected_mul = np.array(data1) * np.array(data2)
        np.testing.assert_allclose(decrypted_mul[:len(expected_mul)], expected_mul, rtol=1e-2)
    
    def test_performance_tracking(self, seal_manager):
        """Test performance metrics collection"""
        op_manager = HomomorphicOperationManager(seal_manager)
        
        # Perform some operations
        data1 = np.random.random(10).tolist()
        data2 = np.random.random(10).tolist()
        
        vec1 = op_manager.encrypt_vector(data1, "performance")
        vec2 = op_manager.encrypt_vector(data2, "performance")
        
        # Perform operations to generate metrics
        for _ in range(5):
            op_manager.homomorphic_add(vec1, vec2)
            op_manager.homomorphic_multiply(vec1, vec2)
        
        # Get performance stats
        stats = op_manager.get_performance_stats()
        
        assert 'add' in stats
        assert 'multiply' in stats
        assert stats['add']['count'] == 5
        assert stats['multiply']['count'] == 5
        assert stats['add']['avg_time'] > 0
        assert stats['multiply']['avg_time'] > 0

class TestAdvancedKeyManager:
    """Test advanced key management functionality"""
    
    @pytest.fixture
    def mock_security_context(self):
        """Mock security context for testing"""
        context = Mock(spec=SecurityContext)
        context.get_user_id.return_value = "test_user"
        context.get_trust_score.return_value = 0.9
        context.get_trust_level.return_value = TrustLevel.TRUSTED
        context.get_context_id.return_value = "test_context"
        context.get_encryption_key.return_value = "test_encryption_key"
        return context
    
    @pytest.fixture
    def temp_key_store(self):
        """Create temporary key store for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def key_manager(self, mock_security_context, temp_key_store):
        """Create key manager instance for testing"""
        return AdvancedKeyManager(mock_security_context, temp_key_store)
    
    def test_key_store_initialization(self, key_manager, temp_key_store):
        """Test key store directory structure creation"""
        assert os.path.exists(f"{temp_key_store}/active")
        assert os.path.exists(f"{temp_key_store}/archived")
        assert os.path.exists(f"{temp_key_store}/audit")
        assert os.path.exists(f"{temp_key_store}/backup")
    
    def test_key_pair_generation(self, key_manager):
        """Test cryptographic key pair generation"""
        # Generate SEAL key pair
        key_id, pub_key_id = key_manager.generate_key_pair(
            KeyType.SEAL_SECRET, 
            "test_context", 
            128
        )
        
        assert key_id is not None
        assert pub_key_id is not None
        assert key_id != pub_key_id
        
        # Generate specialized key
        galois_key_id, _ = key_manager.generate_key_pair(
            KeyType.GALOIS, 
            "test_context", 
            128
        )
        
        assert galois_key_id is not None
    
    def test_key_retrieval(self, key_manager):
        """Test secure key retrieval"""
        # Generate a key
        key_id, _ = key_manager.generate_key_pair(
            KeyType.SEAL_SECRET, 
            "test_context", 
            128
        )
        
        # Retrieve the key
        retrieved_key = key_manager.get_key(key_id, KeyType.SEAL_SECRET)
        
        assert retrieved_key is not None
        assert isinstance(retrieved_key, bytes)
        assert len(retrieved_key) > 0
    
    def test_key_rotation(self, key_manager):
        """Test automatic key rotation"""
        # Generate initial key
        old_key_id, _ = key_manager.generate_key_pair(
            KeyType.SEAL_SECRET, 
            "test_context", 
            128
        )
        
        # Force rotation
        new_key_id = key_manager.rotate_key(old_key_id, force=True)
        
        assert new_key_id != old_key_id
        
        # Verify old key is archived
        with pytest.raises(KeyError):
            key_manager.get_key(old_key_id, KeyType.SEAL_SECRET)
        
        # Verify new key works
        new_key = key_manager.get_key(new_key_id, KeyType.SEAL_SECRET)
        assert new_key is not None
    
    def test_security_controls(self, key_manager, mock_security_context):
        """Test security controls and access restrictions"""
        # Generate a key
        key_id, _ = key_manager.generate_key_pair(
            KeyType.SEAL_SECRET, 
            "test_context", 
            128
        )
        
        # Test with low trust score
        mock_security_context.get_trust_score.return_value = 0.3
        
        with pytest.raises(PermissionError):
            key_manager.get_key(key_id, KeyType.SEAL_SECRET)
        
        # Reset trust score
        mock_security_context.get_trust_score.return_value = 0.9
        
        # Should work now
        key = key_manager.get_key(key_id, KeyType.SEAL_SECRET)
        assert key is not None
    
    def test_key_statistics(self, key_manager):
        """Test key management statistics"""
        # Generate some keys
        for i in range(3):
            key_manager.generate_key_pair(
                KeyType.SEAL_SECRET, 
                f"context_{i}", 
                128
            )
        
        # Get statistics
        stats = key_manager.get_key_statistics()
        
        assert stats['total_keys'] >= 3  # At least 3 secret keys + possible public keys
        assert stats['active_keys'] >= 3
        assert stats['expiring_keys'] == 0
        assert KeyType.SEAL_SECRET.value in stats['key_types']
        assert stats['total_usage'] >= 0
    
    def test_key_persistence(self, key_manager, temp_key_store):
        """Test key persistence across manager instances"""
        # Generate key with first manager instance
        key_id, _ = key_manager.generate_key_pair(
            KeyType.SEAL_SECRET, 
            "test_context", 
            128
        )
        
        # Create new manager instance with same key store
        mock_context = Mock(spec=SecurityContext)
        mock_context.get_user_id.return_value = "test_user"
        mock_context.get_trust_score.return_value = 0.9
        mock_context.get_trust_level.return_value = TrustLevel.TRUSTED
        mock_context.get_context_id.return_value = "test_context"
        mock_context.get_encryption_key.return_value = "test_encryption_key"
        
        new_manager = AdvancedKeyManager(mock_context, temp_key_store)
        
        # Should be able to retrieve the key
        retrieved_key = new_manager.get_key(key_id, KeyType.SEAL_SECRET)
        assert retrieved_key is not None
    
    def test_audit_logging(self, key_manager, temp_key_store):
        """Test audit logging functionality"""
        # Generate and use a key
        key_id, _ = key_manager.generate_key_pair(
            KeyType.SEAL_SECRET, 
            "test_context", 
            128
        )
        
        key_manager.get_key(key_id, KeyType.SEAL_SECRET)
        
        # Check audit log exists
        audit_file = f"{temp_key_store}/audit/key_operations.log"
        assert os.path.exists(audit_file)
        
        # Read and verify audit entries
        with open(audit_file, 'r') as f:
            entries = [line.strip() for line in f if line.strip()]
        
        assert len(entries) >= 2  # At least generate and access operations
        
        # Verify audit entry structure
        import json
        first_entry = json.loads(entries[0])
        assert 'timestamp' in first_entry
        assert 'operation' in first_entry
        assert 'resource' in first_entry
        assert 'user_id' in first_entry
        assert 'details' in first_entry

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
