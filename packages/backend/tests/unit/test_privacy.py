"""
Unit tests for differential privacy module.
Tests Laplace mechanism, budget management, and vault encryption.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
import base64
import os

from ...core.privacy.mechanism import LaplaceNoiseMechanism, PrivacyParameters
from ...core.privacy.models import QueryType, PrivacyLevel, PrivacyQuery, UserPrivacyProfile, VaultCreateRequest
from ...core.privacy.budget import BudgetTracker, PrivacyBudgetManager
from ...core.privacy.vault import FernetKeyManager, EncryptedUserVault, VaultManager


class TestLaplaceNoiseMechanism:
    """Test Laplace noise mechanism implementation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mechanism = LaplaceNoiseMechanism(secure_random=False)
        self.secure_mechanism = LaplaceNoiseMechanism(secure_random=True)
    
    def test_noise_scale_calculation(self):
        """Test noise scale calculation for different epsilon values."""
        test_cases = [
            (1.0, 0.5, 2.0),  # sensitivity=1.0, epsilon=0.5, expected_scale=2.0
            (2.0, 1.0, 2.0),  # sensitivity=2.0, epsilon=1.0, expected_scale=2.0
            (0.5, 0.1, 5.0),  # sensitivity=0.5, epsilon=0.1, expected_scale=5.0
        ]
        
        for sensitivity, epsilon, expected_scale in test_cases:
            scale = self.mechanism.calculate_noise_scale(sensitivity, epsilon)
            assert abs(scale - expected_scale) < 1e-10, f"Expected {expected_scale}, got {scale}"
    
    def test_noise_scale_validation(self):
        """Test noise scale calculation with invalid parameters."""
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            self.mechanism.calculate_noise_scale(1.0, 0.0)
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            self.mechanism.calculate_noise_scale(1.0, -0.5)
        
        with pytest.raises(ValueError, match="Sensitivity must be positive"):
            self.mechanism.calculate_noise_scale(0.0, 1.0)
    
    def test_noise_generation_properties(self):
        """Test statistical properties of generated noise."""
        scale = 1.0
        sample_size = 1000
        
        # Generate noise samples
        noise_samples = [
            self.mechanism.generate_laplace_noise(scale) 
            for _ in range(sample_size)
        ]
        
        # Test statistical properties
        mean_noise = np.mean(noise_samples)
        std_noise = np.std(noise_samples)
        expected_std = np.sqrt(2) * scale
        
        # Mean should be close to 0
        assert abs(mean_noise) < 0.1, f"Mean noise {mean_noise} too far from 0"
        
        # Standard deviation should match Laplace distribution
        assert abs(std_noise - expected_std) / expected_std < 0.1, \
            f"Std deviation mismatch: expected {expected_std}, got {std_noise}"
    
    def test_vector_noise_generation(self):
        """Test vector noise generation."""
        scale = 1.0
        vector_sizes = [1, 10, 100, 1000]
        
        for size in vector_sizes:
            noise_vector = self.mechanism.generate_laplace_noise(scale, size=size)
            
            assert len(noise_vector) == size
            assert all(isinstance(n, (float, np.floating)) for n in noise_vector)
            
            # Check statistical properties for larger vectors
            if size >= 100:
                mean_noise = np.mean(noise_vector)
                assert abs(mean_noise) < 0.2, f"Vector mean too far from 0: {mean_noise}"
    
    def test_add_noise_scalar(self):
        """Test adding noise to scalar values."""
        test_cases = [
            (100.0, PrivacyParameters(epsilon=1.0, sensitivity=1.0)),
            (0.0, PrivacyParameters(epsilon=0.5, sensitivity=2.0)),
            (-50.0, PrivacyParameters(epsilon=2.0, sensitivity=0.5)),
        ]
        
        for true_result, privacy_params in test_cases:
            noisy_result, noise_magnitude = self.mechanism.add_noise(true_result, privacy_params)
            
            assert isinstance(noisy_result, float)
            assert isinstance(noise_magnitude, float)
            assert noise_magnitude >= 0
            
            # Noise should be bounded reasonably (probabilistic test)
            expected_scale = privacy_params.sensitivity / privacy_params.epsilon
            max_reasonable_noise = expected_scale * 10  # 10 standard deviations
            assert abs(noisy_result - true_result) <= max_reasonable_noise
    
    def test_add_noise_vector(self):
        """Test adding noise to vector values."""
        true_result = [10.0, 20.0, 30.0, 40.0, 50.0]
        privacy_params = PrivacyParameters(epsilon=1.0, sensitivity=1.0)
        
        noisy_result, noise_magnitude = self.mechanism.add_noise(true_result, privacy_params)
        
        assert isinstance(noisy_result, list)
        assert len(noisy_result) == len(true_result)
        assert isinstance(noise_magnitude, float)
        assert noise_magnitude > 0
        
        # Each element should be modified
        for original, noisy in zip(true_result, noisy_result):
            assert isinstance(noisy, float)
    
    def test_privacy_parameters_generation(self):
        """Test privacy parameter generation for different query types and levels."""
        test_cases = [
            (QueryType.COUNT, PrivacyLevel.MINIMAL, 1.0, 1.0),
            (QueryType.COUNT, PrivacyLevel.STANDARD, 0.5, 1.0),
            (QueryType.COUNT, PrivacyLevel.HIGH, 0.1, 1.0),
            (QueryType.COUNT, PrivacyLevel.MAXIMUM, 0.01, 1.0),
            (QueryType.SUM, PrivacyLevel.STANDARD, 0.5, 1.0),
            (QueryType.MEAN, PrivacyLevel.HIGH, 0.1, 1.0),
        ]
        
        for query_type, privacy_level, expected_epsilon, expected_sensitivity in test_cases:
            params = self.mechanism.get_privacy_parameters(query_type, privacy_level)
            
            assert params.epsilon == expected_epsilon
            assert params.sensitivity == expected_sensitivity
            assert params.delta == 1e-5
    
    def test_custom_privacy_parameters(self):
        """Test privacy parameters with custom values."""
        custom_epsilon = 0.75
        custom_sensitivity = 2.5
        
        params = self.mechanism.get_privacy_parameters(
            QueryType.COUNT,
            PrivacyLevel.STANDARD,
            custom_epsilon=custom_epsilon,
            custom_sensitivity=custom_sensitivity
        )
        
        assert params.epsilon == custom_epsilon
        assert params.sensitivity == custom_sensitivity
        assert params.delta == 1e-5
    
    def test_accuracy_loss_estimation(self):
        """Test accuracy loss estimation for different confidence levels."""
        privacy_params = PrivacyParameters(epsilon=1.0, sensitivity=1.0)
        
        confidence_levels = [0.90, 0.95, 0.99]
        previous_loss = 0.0
        
        for confidence in confidence_levels:
            accuracy_loss = self.mechanism.estimate_accuracy_loss(privacy_params, confidence)
            
            assert isinstance(accuracy_loss, float)
            assert accuracy_loss > 0
            assert accuracy_loss > previous_loss  # Higher confidence = more noise
            previous_loss = accuracy_loss
    
    def test_privacy_budget_validation(self):
        """Test privacy budget validation logic."""
        validation_cases = [
            (0.5, 1.0, True),   # Requested < Available
            (1.0, 1.0, True),   # Requested = Available
            (1.5, 1.0, False),  # Requested > Available
            (0.0, 1.0, False),  # Invalid requested (zero)
            (0.5, 0.0, False),  # Invalid available (zero)
        ]
        
        for requested, available, expected in validation_cases:
            if requested <= 0 or available <= 0:
                # Should handle invalid inputs gracefully
                result = self.mechanism.validate_privacy_budget(requested, available)
                assert result == expected
            else:
                result = self.mechanism.validate_privacy_budget(requested, available)
                assert result == expected
    
    def test_privacy_loss_composition(self):
        """Test privacy loss composition with various epsilon combinations."""
        test_cases = [
            ([0.1, 0.2, 0.3], 0.6),
            ([1.0], 1.0),
            ([0.5, 0.5, 0.5, 0.5], 2.0),
            ([], 0.0),
        ]
        
        for epsilon_values, expected_total in test_cases:
            total_epsilon = self.mechanism.compose_privacy_loss(epsilon_values)
            assert abs(total_epsilon - expected_total) < 1e-10
    
    def test_optimal_epsilon_calculation(self):
        """Test optimal epsilon calculation for target accuracy."""
        test_cases = [
            (1.0, 1.0, 0.95),  # target_accuracy, sensitivity, confidence
            (2.0, 1.0, 0.90),
            (0.5, 2.0, 0.99),
        ]
        
        for target_accuracy, sensitivity, confidence in test_cases:
            epsilon = self.mechanism.calculate_optimal_epsilon(
                target_accuracy, sensitivity, confidence
            )
            
            assert isinstance(epsilon, float)
            assert 0.01 <= epsilon <= 10.0  # Within reasonable bounds
            
            # Verify the relationship holds approximately
            privacy_params = PrivacyParameters(epsilon=epsilon, sensitivity=sensitivity)
            estimated_loss = self.mechanism.estimate_accuracy_loss(privacy_params, confidence)
            assert abs(estimated_loss - target_accuracy) / target_accuracy < 0.1
    
    def test_secure_vs_pseudo_random(self):
        """Test that secure and pseudo-random generation produce different results."""
        scale = 1.0
        sample_size = 100
        
        # Generate samples with both methods
        secure_samples = [self.secure_mechanism.generate_laplace_noise(scale) for _ in range(sample_size)]
        pseudo_samples = [self.mechanism.generate_laplace_noise(scale) for _ in range(sample_size)]
        
        # Samples should be different (probabilistic test)
        differences = sum(1 for s, p in zip(secure_samples, pseudo_samples) if abs(s - p) > 1e-10)
        assert differences > sample_size * 0.95  # At least 95% should be different
        
        # Both should have reasonable statistical properties
        for samples in [secure_samples, pseudo_samples]:
            mean_val = np.mean(samples)
            assert abs(mean_val) < 0.5  # Mean close to 0


class TestFernetKeyManager:
    """Test Fernet key management functionality."""
    
    def test_key_generation(self):
        """Test Fernet key generation."""
        key1 = FernetKeyManager.generate_key()
        key2 = FernetKeyManager.generate_key()
        
        # Keys should be proper format
        assert isinstance(key1, bytes)
        assert isinstance(key2, bytes)
        assert len(key1) == 44  # Base64 encoded 32-byte key
        assert len(key2) == 44
        
        # Keys should be different
        assert key1 != key2
        
        # Keys should be valid Fernet keys
        from cryptography.fernet import Fernet
        try:
            Fernet(key1)
            Fernet(key2)
        except Exception as e:
            pytest.fail(f"Generated keys are not valid Fernet keys: {e}")
    
    def test_salt_generation(self):
        """Test salt generation."""
        salt1 = FernetKeyManager.generate_salt()
        salt2 = FernetKeyManager.generate_salt()
        
        assert isinstance(salt1, bytes)
        assert isinstance(salt2, bytes)
        assert len(salt1) == 16
        assert len(salt2) == 16
        assert salt1 != salt2  # Should be different
    
    def test_key_derivation_consistency(self):
        """Test that key derivation is consistent with same inputs."""
        password = "test_password_123"
        salt = FernetKeyManager.generate_salt()
        
        # Derive key multiple times
        key1 = FernetKeyManager.derive_key_from_password(password, salt)
        key2 = FernetKeyManager.derive_key_from_password(password, salt)
        key3 = FernetKeyManager.derive_key_from_password(password, salt)
        
        # All keys should be identical
        assert key1 == key2 == key3
        assert isinstance(key1, bytes)
        assert len(key1) == 44
    
    def test_key_derivation_different_passwords(self):
        """Test that different passwords produce different keys."""
        salt = FernetKeyManager.generate_salt()
        
        passwords = ["password1", "password2", "different_password", ""]
        keys = []
        
        for password in passwords:
            key = FernetKeyManager.derive_key_from_password(password, salt)
            keys.append(key)
        
        # All keys should be different
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert keys[i] != keys[j], f"Keys for passwords {i} and {j} are identical"
    
    def test_key_derivation_different_salts(self):
        """Test that different salts produce different keys."""
        password = "consistent_password"
        
        salts = [FernetKeyManager.generate_salt() for _ in range(5)]
        keys = [FernetKeyManager.derive_key_from_password(password, salt) for salt in salts]
        
        # All keys should be different
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert keys[i] != keys[j], f"Keys with salts {i} and {j} are identical"
    
    def test_key_identifier_generation(self):
        """Test key identifier generation."""
        key1 = FernetKeyManager.generate_key()
        key2 = FernetKeyManager.generate_key()
        
        id1 = FernetKeyManager.hash_key_identifier(key1)
        id2 = FernetKeyManager.hash_key_identifier(key2)
        id3 = FernetKeyManager.hash_key_identifier(key1)  # Same key again
        
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert len(id1) == 16
        assert len(id2) == 16
        assert id1 != id2  # Different keys should have different identifiers
        assert id1 == id3  # Same key should have same identifier
    
    def test_key_derivation_performance(self):
        """Test that key derivation takes reasonable time (security check)."""
        import time
        
        password = "test_password"
        salt = FernetKeyManager.generate_salt()
        
        start_time = time.time()
        key = FernetKeyManager.derive_key_from_password(password, salt)
        derivation_time = time.time() - start_time
        
        # Should take at least 50ms (security requirement) but less than 2 seconds
        assert 0.05 <= derivation_time <= 2.0, f"Key derivation took {derivation_time:.3f}s"
        assert isinstance(key, bytes)
        assert len(key) == 44


class TestEncryptedUserVault:
    """Test encrypted user vault functionality."""
    
    def setup_method(self):
        """Set up test vault."""
        self.user_id = "test-user-123"
        self.key = FernetKeyManager.generate_key()
        self.vault = EncryptedUserVault(self.user_id, self.key)
    
    def test_vault_initialization(self):
        """Test vault initialization."""
        assert self.vault.user_id == self.user_id
        assert hasattr(self.vault, 'fernet')
        assert hasattr(self.vault, 'key_identifier')
        assert isinstance(self.vault.key_identifier, str)
        assert len(self.vault.key_identifier) == 16
    
    def test_data_encryption_decryption_simple(self):
        """Test basic encryption and decryption."""
        test_data = {
            "username": "testuser",
            "password": "secretpass",
            "notes": "This is a test entry"
        }
        
        # Encrypt data
        encrypted = self.vault.encrypt_data(test_data)
        assert isinstance(encrypted, str)
        assert len(encrypted) > 0
        assert encrypted != str(test_data)
        
        # Decrypt data
        decrypted = self.vault.decrypt_data(encrypted)
        assert decrypted == test_data
    
    def test_data_encryption_decryption_complex(self):
        """Test encryption/decryption with complex data structures."""
        complex_data = {
            "user_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "preferences": {
                    "theme": "dark",
                    "language": "en",
                    "notifications": True
                }
            },
            "api_keys": [
                {"service": "github", "key": "ghp_1234567890"},
                {"service": "aws", "key": "AKIA1234567890"}
            ],
            "metadata": {
                "created": "2024-01-01T00:00:00Z",
                "version": 2,
                "tags": ["personal", "work", "important"]
            },
            "numbers": [1, 2, 3.14159, -42],
            "boolean_flags": {"active": True, "verified": False}
        }
        
        encrypted = self.vault.encrypt_data(complex_data)
        decrypted = self.vault.decrypt_data(encrypted)
        assert decrypted == complex_data
    
    def test_encryption_determinism(self):
        """Test that encryption is non-deterministic (includes timestamp/nonce)."""
        test_data = {"test": "data"}
        
        encrypted1 = self.vault.encrypt_data(test_data)
        encrypted2 = self.vault.encrypt_data(test_data)
        
        # Encrypted results should be different (Fernet includes timestamp)
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same data
        assert self.vault.decrypt_data(encrypted1) == test_data
        assert self.vault.decrypt_data(encrypted2) == test_data
    
    def test_vault_entry_creation(self):
        """Test vault entry creation."""
        request = VaultCreateRequest(
            title="Test Credentials",
            data={"username": "test", "password": "secret", "url": "https://example.com"},
            data_type="credentials",
            tags=["test", "demo", "web"]
        )
        
        entry = self.vault.encrypt_vault_entry(request)
        
        assert entry.user_id == self.user_id
        assert entry.title == "Test Credentials"
        assert entry.data_type == "credentials"
        assert entry.tags == ["test", "demo", "web"]
        assert entry.encrypted_data != ""
        assert isinstance(entry.entry_id, str)
        assert len(entry.entry_id) > 0
    
    def test_vault_entry_decryption(self):
        """Test vault entry decryption."""
        original_data = {
            "username": "admin",
            "password": "super_secret_123",
            "notes": "Production database credentials",
            "host": "prod-db.example.com",
            "port": 5432
        }
        
        request = VaultCreateRequest(
            title="Database Credentials",
            data=original_data,
            data_type="credentials"
        )
        
        entry = self.vault.encrypt_vault_entry(request)
        decrypted_data = self.vault.decrypt_vault_entry(entry)
        
        assert decrypted_data == original_data
    
    def test_vault_entry_update(self):
        """Test vault entry updates."""
        # Create initial entry
        initial_data = {"username": "user", "password": "old_pass"}
        request = VaultCreateRequest(
            title="Initial Entry",
            data=initial_data,
            data_type="credentials"
        )
        entry = self.vault.encrypt_vault_entry(request)
        
        # Update the entry
        updated_data = {"username": "user", "password": "new_secure_pass", "notes": "Updated password"}
        from ...core.privacy.models import VaultUpdateRequest
        update_request = VaultUpdateRequest(
            title="Updated Entry",
            data=updated_data,
            tags=["updated", "secure"]
        )
        
        updated_entry = self.vault.update_vault_entry(entry, update_request)
        
        # Verify updates
        assert updated_entry.title == "Updated Entry"
        assert updated_entry.tags == ["updated", "secure"]
        
        # Verify encrypted data was updated
        decrypted_data = self.vault.decrypt_vault_entry(updated_entry)
        assert decrypted_data == updated_data
    
    def test_invalid_decryption(self):
        """Test decryption with invalid data."""
        with pytest.raises(Exception):  # Should raise Fernet decryption error
            self.vault.decrypt_data("invalid_encrypted_data")
        
        with pytest.raises(Exception):  # Should raise JSON decode error
            self.vault.decrypt_data("dmFsaWRfYmFzZTY0X2J1dF9ub3RfdmFsaWRfZmVybmV0")
    
    def test_cross_vault_decryption_fails(self):
        """Test that data encrypted with one key cannot be decrypted with another."""
        # Create second vault with different key
        other_key = FernetKeyManager.generate_key()
        other_vault = EncryptedUserVault("other-user", other_key)
        
        test_data = {"secret": "data"}
        
        # Encrypt with first vault
        encrypted = self.vault.encrypt_data(test_data)
        
        # Try to decrypt with second vault (should fail)
        with pytest.raises(Exception):
            other_vault.decrypt_data(encrypted)


@pytest.mark.asyncio
class TestBudgetTracker:
    """Test privacy budget tracking functionality."""
    
    def setup_method(self):
        """Set up test budget tracker with mocked dependencies."""
        self.tracker = BudgetTracker()
        
        # Mock Redis client
        self.mock_redis = Mock()
        self.tracker.redis_client = self.mock_redis
        
        # Mock Supabase client
        self.mock_supabase = Mock()
        self.tracker.supabase = self.mock_supabase
    
    async def test_default_profile_creation(self):
        """Test creation of default privacy profile."""
        user_id = "new-user-123"
        
        # Mock no existing profile in Redis or Supabase
        self.mock_redis.get.return_value = None
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        
        profile = await self.tracker.get_user_profile(user_id)
        
        # Verify default profile
        assert profile.user_id == user_id
        assert profile.total_epsilon_budget == 10.0
        assert profile.daily_epsilon_limit == 2.0
        assert profile.epsilon_used_today == 0.0
        assert profile.epsilon_used_total == 0.0
        assert profile.preferred_privacy_level == "standard"
        assert isinstance(profile.created_at, datetime)
    
    async def test_cached_profile_retrieval(self):
        """Test retrieval of cached profile from Redis."""
        user_id = "cached-user"
        
        # Mock cached profile data
        cached_profile = {
            "user_id": user_id,
            "total_epsilon_budget": 15.0,
            "daily_epsilon_limit": 3.0,
            "epsilon_used_today": 1.5,
            "epsilon_used_total": 7.2,
            "last_reset_date": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "preferred_privacy_level": "high"
        }
        
        self.mock_redis.get.return_value = json.dumps(cached_profile)
        
        profile = await self.tracker.get_user_profile(user_id)
        
        assert profile.user_id == user_id
        assert profile.total_epsilon_budget == 15.0
        assert profile.daily_epsilon_limit == 3.0
        assert profile.epsilon_used_today == 1.5
        assert profile.epsilon_used_total == 7.2
    
    async def test_daily_budget_reset_needed(self):
        """Test detection and execution of daily budget reset."""
        user_id = "reset-user"
        
        # Mock profile from yesterday
        yesterday = datetime.utcnow() - timedelta(days=1)
        old_profile_data = {
            "user_id": user_id,
            "total_epsilon_budget": 10.0,
            "daily_epsilon_limit": 2.0,
            "epsilon_used_today": 1.8,  # High usage from yesterday
            "epsilon_used_total": 5.0,
            "last_reset_date": yesterday.isoformat(),
            "created_at": yesterday.isoformat(),
            "updated_at": yesterday.isoformat(),
            "preferred_privacy_level": "standard"
        }
        
        self.mock_redis.get.return_value = None
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [old_profile_data]
        
        # Mock the database update for reset
        self.mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{
            **old_profile_data,
            "epsilon_used_today": 0.0,
            "last_reset_date": datetime.utcnow().isoformat()
        }]
        
        profile = await self.tracker.get_user_profile(user_id)
        
        # Should have reset daily usage
        assert profile.epsilon_used_today == 0.0
        assert profile.epsilon_used_total == 5.0  # Total should remain
    
    async def test_budget_availability_sufficient(self):
        """Test budget availability check with sufficient budget."""
        user_id = "sufficient-user"
        
        # Mock profile with available budget
        profile_data = {
            "user_id": user_id,
            "total_epsilon_budget": 10.0,
            "daily_epsilon_limit": 2.0,
            "epsilon_used_today": 0.5,
            "epsilon_used_total": 3.0,
            "last_reset_date": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "preferred_privacy_level": "standard"
        }
        
        self.mock_redis.get.return_value = None
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [profile_data]
        
        available, reason, remaining = await self.tracker.check_budget_availability(user_id, 0.3)
        
        assert available is True
        assert remaining == 1.2  # 2.0 daily limit - 0.5 used - 0.3 requested
        assert "available" in reason.lower()
    
    async def test_budget_availability_daily_exceeded(self):
        """Test budget availability check with daily limit exceeded."""
        user_id = "daily-exceeded-user"
        
        # Mock profile with daily budget nearly exhausted
        profile_data = {
            "user_id": user_id,
            "total_epsilon_budget": 10.0,
            "daily_epsilon_limit": 2.0,
            "epsilon_used_today": 1.8,
            "epsilon_used_total": 3.0,
            "last_reset_date": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "preferred_privacy_level": "standard"
        }
        
        self.mock_redis.get.return_value = None
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [profile_data]
        
        available, reason, remaining = await self.tracker.check_budget_availability(user_id, 0.5)
        
        assert available is False
        assert "daily" in reason.lower()
        assert remaining == 0.2  # 2.0 - 1.8
    
    async def test_budget_availability_total_exceeded(self):
        """Test budget availability check with total budget exceeded."""
        user_id = "total-exceeded-user"
        
        # Mock profile with total budget nearly exhausted
        profile_data = {
            "user_id": user_id,
            "total_epsilon_budget": 10.0,
            "daily_epsilon_limit": 2.0,
            "epsilon_used_today": 0.5,
            "epsilon_used_total": 9.8,
            "last_reset_date": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "preferred_privacy_level": "standard"
        }
        
        self.mock_redis.get.return_value = None
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [profile_data]
        
        available, reason, remaining = await self.tracker.check_budget_availability(user_id, 0.5)
        
        assert available is False
        assert "total" in reason.lower()
        assert remaining == 0.2  # 10.0 - 9.8
    
    async def test_budget_consumption_success(self):
        """Test successful budget consumption."""
        user_id = "consume-user"
        epsilon_consumed = 0.3
        query_id = "query-123"
        
        # Mock successful Redis operations
        self.mock_redis.hincrby.return_value = None
        self.mock_redis.hset.return_value = None
        self.mock_redis.expire.return_value = None
        
        # Mock successful Supabase update
        self.mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{"updated": True}]
        
        result = await self.tracker.consume_budget(user_id, epsilon_consumed, query_id)
        
        assert result is True
        
        # Verify Redis calls
        assert self.mock_redis.hincrby.called
        assert self.mock_supabase.table.return_value.update.called
    
    async def test_audit_log_creation(self):
        """Test privacy query audit log creation."""
        user_id = "audit-user"
        operation_type = "query_executed"
        epsilon_consumed = 0.5
        query_id = "query-456"
        metadata = {"query_type": "count", "dataset": "test"}
        
        # Mock successful audit log insertion
        self.mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [{"log_id": "log-123"}]
        
        await self.tracker._create_audit_log(user_id, operation_type, epsilon_consumed, query_id, metadata)
        
        # Verify Supabase insert was called
        self.mock_supabase.table.return_value.insert.assert_called()
        
        # Get the call arguments to verify structure
        call_args = self.mock_supabase.table.return_value.insert.call_args[0][0]
        assert call_args["user_id"] == user_id
        assert call_args["operation_type"] == operation_type
        assert call_args["epsilon_consumed"] == epsilon_consumed
        assert call_args["query_id"] == query_id
        assert call_args["metadata"] == metadata


@pytest.mark.asyncio
class TestPrivacyBudgetManager:
    """Test high-level privacy budget management."""
    
    def setup_method(self):
        """Set up test budget manager."""
        self.manager = PrivacyBudgetManager()
        self.manager.budget_tracker = Mock()
        self.manager.mechanism = Mock()
    
    async def test_privacy_budget_request_approved(self):
        """Test successful privacy budget request."""
        user_id = "request-user"
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.COUNT,
            dataset_id="test-dataset",
            privacy_level=PrivacyLevel.STANDARD
        )
        
        # Mock privacy parameters
        self.manager.mechanism.get_privacy_parameters.return_value = PrivacyParameters(
            epsilon=0.5, sensitivity=1.0, delta=1e-5
        )
        
        # Mock budget availability
        self.manager.budget_tracker.check_budget_availability.return_value = (True, "Budget available", 1.5)
        
        approved, reason, epsilon_allocated = await self.manager.request_privacy_budget(query)
        
        assert approved is True
        assert epsilon_allocated == 0.5
        assert "approved" in reason.lower()
    
    async def test_privacy_budget_request_denied(self):
        """Test privacy budget request denial."""
        user_id = "denied-user"
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.MEAN,
            dataset_id="test-dataset",
            privacy_level=PrivacyLevel.HIGH
        )
        
        # Mock privacy parameters
        self.manager.mechanism.get_privacy_parameters.return_value = PrivacyParameters(
            epsilon=0.1, sensitivity=1.0, delta=1e-5
        )
        
        # Mock budget exhaustion
        self.manager.budget_tracker.check_budget_availability.return_value = (
            False, "Daily budget exceeded", 0.05
        )
        
        approved, reason, epsilon_allocated = await self.manager.request_privacy_budget(query)
        
        assert approved is False
        assert epsilon_allocated is None
        assert "exceeded" in reason.lower()
    
    async def test_execute_private_query_success(self):
        """Test successful private query execution."""
        user_id = "execute-user"
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.COUNT,
            dataset_id="test-dataset",
            privacy_level=PrivacyLevel.STANDARD
        )
        true_result = 1000.0
        
        # Mock successful budget request
        self.manager.request_privacy_budget = AsyncMock(return_value=(True, "Approved", 0.5))
        
        # Mock privacy parameters and noise addition
        privacy_params = PrivacyParameters(epsilon=0.5, sensitivity=1.0, delta=1e-5)
        self.manager.mechanism.get_privacy_parameters.return_value = privacy_params
        self.manager.mechanism.add_noise.return_value = (1002.3, 2.3)
        self.manager.mechanism.estimate_accuracy_loss.return_value = 1.96
        
        # Mock budget consumption and profile update
        mock_profile = Mock()
        mock_profile.daily_epsilon_limit = 2.0
        mock_profile.epsilon_used_today = 1.0
        mock_profile.total_epsilon_budget = 10.0
        mock_profile.epsilon_used_total = 5.5
        
        self.manager.budget_tracker.consume_budget.return_value = mock_profile
        
        result = await self.manager.execute_private_query(query, true_result)
        
        assert result["success"] is True
        assert result["query_id"] == query.query_id
        assert result["result"] == 1002.3
        assert result["epsilon_used"] == 0.5
        assert result["noise_magnitude"] == 2.3
        assert result["accuracy_loss"] == 1.96
        assert "remaining_daily_budget" in result
        assert "remaining_total_budget" in result
    
    async def test_execute_private_query_budget_failure(self):
        """Test private query execution with budget failure."""
        user_id = "budget-fail-user"
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.SUM,
            dataset_id="test-dataset",
            privacy_level=PrivacyLevel.MINIMAL
        )
        true_result = 500.0
        
        # Mock budget request failure
        self.manager.request_privacy_budget = AsyncMock(
            return_value=(False, "Total budget exceeded", None)
        )
        
        result = await self.manager.execute_private_query(query, true_result)
        
        assert result["success"] is False
        assert "budget" in result["error"].lower()
        assert result["query_id"] == query.query_id
    
    async def test_get_user_budget_status(self):
        """Test getting user budget status."""
        user_id = "status-user"
        
        # Mock user profile
        mock_profile = UserPrivacyProfile(
            user_id=user_id,
            total_epsilon_budget=10.0,
            daily_epsilon_limit=2.0,
            epsilon_used_today=1.2,
            epsilon_used_total=6.8,
            preferred_privacy_level="high"
        )
        
        self.manager.budget_tracker.get_user_profile.return_value = mock_profile
        
        status = await self.manager.get_user_budget_status(user_id)
        
        assert status["user_id"] == user_id
        assert status["daily_budget"]["limit"] == 2.0
        assert status["daily_budget"]["used"] == 1.2
        assert status["daily_budget"]["remaining"] == 0.8
        assert status["daily_budget"]["utilization"] == 60.0
        assert status["total_budget"]["limit"] == 10.0
        assert status["total_budget"]["used"] == 6.8
        assert status["total_budget"]["remaining"] == 3.2
        assert status["total_budget"]["utilization"] == 68.0
        assert status["preferred_privacy_level"] == "high"
    
    async def test_update_user_preferences_success(self):
        """Test successful user preference updates."""
        user_id = "pref-user"
        preferences = {
            "preferred_privacy_level": "maximum",
            "daily_epsilon_limit": 1.5
        }
        
        # Mock successful profile retrieval and update
        mock_profile = UserPrivacyProfile(
            user_id=user_id,
            total_epsilon_budget=10.0,
            daily_epsilon_limit=2.0,
            epsilon_used_today=0.5,
            epsilon_used_total=3.0
        )
        
        self.manager.budget_tracker.get_user_profile.return_value = mock_profile
        self.manager.budget_tracker._save_profile = AsyncMock()
        self.manager.budget_tracker._cache_profile = AsyncMock()
        
        result = await self.manager.update_user_preferences(user_id, preferences)
        
        assert result is True
    
    async def test_update_user_preferences_validation_error(self):
        """Test user preference update with validation error."""
        user_id = "invalid-pref-user"
        preferences = {
            "daily_epsilon_limit": 15.0  # Exceeds total budget of 10.0
        }
        
        # Mock profile with lower total budget
        mock_profile = UserPrivacyProfile(
            user_id=user_id,
            total_epsilon_budget=10.0,
            daily_epsilon_limit=2.0
        )
        
        self.manager.budget_tracker.get_user_profile.return_value = mock_profile
        
        result = await self.manager.update_user_preferences(user_id, preferences)
        
        assert result is False


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_privacy_workflow(self):
        """Test a complete privacy-preserving workflow."""
        # This would be a comprehensive integration test
        # For now, we'll test the component interactions
        
        user_id = "integration-user"
        vault_password = "secure_vault_password_123"
        
        # Test 1: Create vault entry
        key = FernetKeyManager.generate_key()
        vault = EncryptedUserVault(user_id, key)
        
        sensitive_data = {
            "database_url": "postgresql://user:pass@localhost/db",
            "api_key": "sk-1234567890abcdef",
            "privacy_settings": {
                "preferred_epsilon": 0.1,
                "daily_limit": 1.0
            }
        }
        
        vault_request = VaultCreateRequest(
            title="Production Secrets",
            data=sensitive_data,
            data_type="credentials",
            tags=["production", "database", "api"]
        )
        
        vault_entry = vault.encrypt_vault_entry(vault_request)
        
        # Test 2: Verify vault encryption worked
        decrypted_data = vault.decrypt_vault_entry(vault_entry)
        assert decrypted_data == sensitive_data
        
        # Test 3: Test privacy mechanism
        mechanism = LaplaceNoiseMechanism(secure_random=False)  # Use deterministic for testing
        
        true_count = 10000
        privacy_params = mechanism.get_privacy_parameters(QueryType.COUNT, PrivacyLevel.HIGH)
        
        noisy_count, noise_magnitude = mechanism.add_noise(true_count, privacy_params)
        
        # Verify privacy properties
        assert isinstance(noisy_count, float)
        assert noisy_count != true_count  # Should have noise
        assert noise_magnitude > 0
        
        # Test 4: Verify privacy parameters are reasonable
        assert privacy_params.epsilon == 0.1  # High privacy level
        assert privacy_params.sensitivity == 1.0  # Count sensitivity
        
        accuracy_loss = mechanism.estimate_accuracy_loss(privacy_params)
        assert accuracy_loss > 0
    
    def test_performance_benchmarks(self):
        """Test that operations meet performance requirements."""
        import time
        
        # Benchmark 1: Key derivation (should be slow for security)
        start_time = time.time()
        salt = FernetKeyManager.generate_salt()
        key = FernetKeyManager.derive_key_from_password("password", salt)
        key_derivation_time = time.time() - start_time
        
        assert 0.05 <= key_derivation_time <= 2.0  # 50ms to 2s is acceptable
        
        # Benchmark 2: Encryption (should be fast)
        vault = EncryptedUserVault("user", key)
        test_data = {"key": "value", "number": 42}
        
        start_time = time.time()
        encrypted = vault.encrypt_data(test_data)
        encryption_time = time.time() - start_time
        
        assert encryption_time < 0.01  # Should be under 10ms
        
        # Benchmark 3: Noise generation (should be fast)
        mechanism = LaplaceNoiseMechanism(secure_random=False)
        
        start_time = time.time()
        for _ in range(1000):
            mechanism.generate_laplace_noise(1.0)
        noise_time = time.time() - start_time
        
        assert noise_time < 0.1  # 1000 operations in under 100ms
    
    def test_security_properties(self):
        """Test security properties of the implementation."""
        # Test 1: Different keys produce different ciphertexts
        key1 = FernetKeyManager.generate_key()
        key2 = FernetKeyManager.generate_key()
        
        vault1 = EncryptedUserVault("user1", key1)
        vault2 = EncryptedUserVault("user2", key2)
        
        test_data = {"secret": "information"}
        
        encrypted1 = vault1.encrypt_data(test_data)
        encrypted2 = vault2.encrypt_data(test_data)
        
        assert encrypted1 != encrypted2
        
        # Test 2: Cross-vault decryption should fail
        with pytest.raises(Exception):
            vault1.decrypt_data(encrypted2)
        
        # Test 3: Privacy mechanism should add significant noise for small epsilon
        mechanism = LaplaceNoiseMechanism(secure_random=False)
        privacy_params = PrivacyParameters(epsilon=0.01, sensitivity=1.0)  # Very private
        
        true_value = 100.0
        noisy_values = []
        
        for _ in range(10):
            noisy_val, _ = mechanism.add_noise(true_value, privacy_params)
            noisy_values.append(noisy_val)
        
        # With very small epsilon, noise should be substantial
        variance = np.var(noisy_values)
        assert variance > 1000  # High variance indicates significant noise


# Performance and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for the privacy system."""
    
    def test_large_data_encryption(self):
        """Test encryption performance with large data."""
        key = FernetKeyManager.generate_key()
        vault = EncryptedUserVault("perf-user", key)
        
        # Create large data structure
        large_data = {
            "large_list": list(range(10000)),
            "text_data": "x" * 50000,  # 50KB string
            "nested_dict": {f"key_{i}": f"value_{i}" for i in range(1000)}
        }
        
        import time
        start_time = time.time()
        encrypted = vault.encrypt_data(large_data)
        encryption_time = time.time() - start_time
        
        start_time = time.time()
        decrypted = vault.decrypt_data(encrypted)
        decryption_time = time.time() - start_time
        
        # Performance assertions
        assert encryption_time < 1.0  # Under 1 second
        assert decryption_time < 1.0  # Under 1 second
        assert decrypted == large_data  # Data integrity
    
    def test_concurrent_noise_generation(self):
        """Test noise generation under concurrent load."""
        import concurrent.futures
        import threading
        
        mechanism = LaplaceNoiseMechanism(secure_random=True)
        scale = 1.0
        num_threads = 5
        operations_per_thread = 200
        
        def generate_noise_batch():
            thread_id = threading.current_thread().ident
            results = []
            for i in range(operations_per_thread):
                noise = mechanism.generate_laplace_noise(scale)
                results.append((thread_id, i, noise))
            return results
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(generate_noise_batch) for _ in range(num_threads)]
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
        
        # Verify results
        assert len(all_results) == num_threads * operations_per_thread
        
        # Check that different threads produced different results
        thread_results = {}
        for thread_id, op_id, noise in all_results:
            if thread_id not in thread_results:
                thread_results[thread_id] = []
            thread_results[thread_id].append(noise)
        
        assert len(thread_results) == num_threads
        
        # Statistical check - noise should have reasonable distribution
        all_noise_values = [noise for _, _, noise in all_results]
        mean_noise = np.mean(all_noise_values)
        std_noise = np.std(all_noise_values)
        
        assert abs(mean_noise) < 0.2  # Mean close to 0
        assert 1.0 < std_noise < 2.0  # Reasonable standard deviation


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
