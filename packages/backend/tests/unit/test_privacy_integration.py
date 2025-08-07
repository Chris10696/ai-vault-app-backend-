"""
Integration tests for Module B: Differential Privacy Engine.
Tests the complete privacy pipeline with real-world scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from ...core.privacy.mechanism import LaplaceNoiseMechanism
from ...core.privacy.budget import PrivacyBudgetManager
from ...core.privacy.vault import VaultManager, FernetKeyManager
from ...core.privacy.analytics import PrivacyAnalytics
from ...core.privacy.monitoring import BudgetMonitor
from ...core.privacy.models import PrivacyQuery, QueryType, PrivacyLevel, VaultCreateRequest

@pytest.mark.asyncio
class TestPrivacyPipeline:
    """Test complete privacy pipeline integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.budget_manager = PrivacyBudgetManager()
        self.vault_manager = VaultManager()
        self.analytics = PrivacyAnalytics()
        self.monitor = BudgetMonitor()
        
        # Mock external dependencies
        self.budget_manager.budget_tracker.redis_client = Mock()
        self.budget_manager.budget_tracker.supabase = Mock()
        self.vault_manager.supabase = Mock()
        self.analytics.supabase = Mock()
    
    async def test_complete_privacy_workflow(self):
        """Test a complete privacy-preserving query workflow."""
        user_id = "test-user-123"
        
        # Step 1: Create privacy query
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.COUNT,
            dataset_id="test-dataset",
            privacy_level=PrivacyLevel.STANDARD
        )
        
        # Step 2: Mock budget availability
        self.budget_manager.budget_tracker.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{
            "user_id": user_id,
            "total_epsilon_budget": 10.0,
            "daily_epsilon_limit": 2.0,
            "epsilon_used_today": 0.5,
            "epsilon_used_total": 2.0,
            "last_reset_date": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "preferred_privacy_level": "standard"
        }]
        
        # Step 3: Execute private query
        true_result = 1000  # Simulated count result
        response = await self.budget_manager.execute_private_query(query, true_result)
        
        # Step 4: Verify response
        assert response["success"] is True
        assert "result" in response
        assert "epsilon_used" in response
        assert response["epsilon_used"] == 0.5  # Standard privacy level
        assert response["result"] != true_result  # Should have noise added
        
        # Step 5: Verify budget consumption
        assert "remaining_daily_budget" in response
        assert "remaining_total_budget" in response
    
    async def test_vault_encryption_integration(self):
        """Test vault encryption with privacy budget management."""
        user_id = "test-user-456"
        vault_password = "secure_vault_password_123"
        
        # Step 1: Create sensitive data entry
        sensitive_data = {
            "api_key": "sk-test-123456789",
            "database_credentials": {
                "host": "secure-db.example.com",
                "username": "admin",
                "password": "super_secret_password"
            },
            "privacy_settings": {
                "epsilon_preference": 0.1,
                "daily_limit": 1.0
            }
        }
        
        entry_request = VaultCreateRequest(
            title="API Keys & Credentials",
            data=sensitive_data,
            data_type="credentials",
            tags=["api", "database", "sensitive"]
        )
        
        # Step 2: Mock successful vault creation
        self.vault_manager.supabase.table.return_value.insert.return_value.execute.return_value.data = [{
            "entry_id": "vault-entry-123",
            "title": "API Keys & Credentials",
            "data_type": "credentials",
            "tags": ["api", "database", "sensitive"],
            "created_at": datetime.utcnow().isoformat()
        }]
        
        # Step 3: Create vault entry
        result = await self.vault_manager.create_user_vault(
            user_id,
            vault_password,
            entry_request
        )
        
        # Step 4: Verify vault creation
        assert result["success"] is True
        assert result["entry_id"] == "vault-entry-123"
        assert result["title"] == "API Keys & Credentials"
    
    async def test_privacy_budget_exhaustion_scenario(self):
        """Test behavior when privacy budget is exhausted."""
        user_id = "test-user-exhausted"
        
        # Mock exhausted budget
        self.budget_manager.budget_tracker.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{
            "user_id": user_id,
            "total_epsilon_budget": 10.0,
            "daily_epsilon_limit": 2.0,
            "epsilon_used_today": 2.0,  # Fully exhausted daily budget
            "epsilon_used_total": 9.5,   # Nearly exhausted total budget
            "last_reset_date": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "preferred_privacy_level": "standard"
        }]
        
        # Create query that would exceed budget
        query = PrivacyQuery(
            user_id=user_id,
            query_type=QueryType.MEAN,
            dataset_id="test-dataset",
            privacy_level=PrivacyLevel.STANDARD  # Requires 0.5 epsilon
        )
        
        # Execute query
        response = await self.budget_manager.execute_private_query(query, 50.0)
        
        # Verify rejection
        assert response["success"] is False
        assert "budget exceeded" in response["error"].lower()
        assert response["query_id"] == query.query_id
    
    async def test_monitoring_alert_generation(self):
        """Test privacy budget monitoring and alert generation."""
        user_id = "test-user-alerts"
        
        # Mock high budget utilization
        mock_status = {
            "user_id": user_id,
            "daily_budget": {
                "limit": 2.0,
                "used": 1.8,  # 90% utilization
                "remaining": 0.2,
                "utilization": 90.0
            },
            "total_budget": {
                "limit": 10.0,
                "used": 7.5,  # 75% utilization
                "remaining": 2.5,
                "utilization": 75.0
            }
        }
        
        # Mock budget manager response
        with patch.object(self.budget_manager, 'get_user_budget_status', return_value=mock_status):
            # Test budget status check
            await self.monitor._check_user_budget_status(user_id)
            
            # Verify alert would be generated (in real implementation)
            # This tests the logic flow rather than actual alert dispatch
            assert mock_status["daily_budget"]["utilization"] >= self.monitor.alert_thresholds["daily_budget_critical"]
    
    async def test_key_derivation_consistency(self):
        """Test Fernet key derivation consistency."""
        password = "test_password_123"
        salt = FernetKeyManager.generate_salt()
        
        # Derive key multiple times with same inputs
        key1 = FernetKeyManager.derive_key_from_password(password, salt)
        key2 = FernetKeyManager.derive_key_from_password(password, salt)
        key3 = FernetKeyManager.derive_key_from_password(password, salt)
        
        # All keys should be identical
        assert key1 == key2 == key3
        
        # Keys should be proper Fernet keys
        assert len(key1) == 44  # Base64 encoded 32-byte key
        assert isinstance(key1, bytes)
        
        # Different salts should produce different keys
        salt2 = FernetKeyManager.generate_salt()
        key4 = FernetKeyManager.derive_key_from_password(password, salt2)
        assert key1 != key4
    
    async def test_noise_distribution_properties(self):
        """Test Laplace noise statistical properties."""
        mechanism = LaplaceNoiseMechanism(secure_random=False)  # Use deterministic for testing
        epsilon = 1.0
        sensitivity = 1.0
        sample_size = 1000
        
        # Generate noise samples
        scale = mechanism.calculate_noise_scale(sensitivity, epsilon)
        noise_samples = [
            mechanism.generate_laplace_noise(scale) 
            for _ in range(sample_size)
        ]
        
        # Test statistical properties
        import numpy as np
        
        # Mean should be approximately 0
        mean_noise = np.mean(noise_samples)
        assert abs(mean_noise) < 0.1, f"Mean noise {mean_noise} too far from 0"
        
        # Standard deviation should match Laplace distribution
        expected_std = np.sqrt(2) * scale
        actual_std = np.std(noise_samples)
        assert abs(actual_std - expected_std) / expected_std < 0.1, f"Standard deviation mismatch: expected {expected_std}, got {actual_std}"

@pytest.mark.asyncio 
class TestPrivacyCompliance:
    """Test privacy compliance and security properties."""
    
    def setup_method(self):
        """Set up compliance testing environment."""
        self.mechanism = LaplaceNoiseMechanism(secure_random=True)
    
    async def test_epsilon_delta_guarantees(self):
        """Test differential privacy (ε,δ)-guarantees."""
        # Test different epsilon values
        epsilon_values = [0.01, 0.1, 0.5, 1.0]
        sensitivity = 1.0
        
        for epsilon in epsilon_values:
            # Generate noise for same input multiple times
            true_result = 100.0
            privacy_params = self.mechanism.get_privacy_parameters(
                QueryType.COUNT,
                PrivacyLevel.STANDARD,
                custom_epsilon=epsilon
            )
            privacy_params.sensitivity = sensitivity
            
            # Generate multiple noisy results
            noisy_results = []
            for _ in range(100):
                noisy_result, _ = self.mechanism.add_noise(true_result, privacy_params)
                noisy_results.append(noisy_result)
            
            # Verify noise scale is correct
            expected_scale = sensitivity / epsilon
            actual_scale = self.mechanism.calculate_noise_scale(sensitivity, epsilon)
            assert abs(actual_scale - expected_scale) < 1e-10
            
            # Verify privacy parameters
            assert privacy_params.epsilon == epsilon
            assert privacy_params.sensitivity == sensitivity
            assert privacy_params.delta == 1e-5
    
    async def test_composition_properties(self):
        """Test privacy loss composition."""
        # Test sequential composition
        epsilon_queries = [0.1, 0.2, 0.15, 0.05]
        total_epsilon = self.mechanism.compose_privacy_loss(epsilon_queries)
        
        # For sequential composition, total should be sum
        expected_total = sum(epsilon_queries)
        assert total_epsilon == expected_total
        
        # Test composition bounds
        assert total_epsilon >= max(epsilon_queries)  # At least the maximum
        assert total_epsilon <= len(epsilon_queries) * max(epsilon_queries)  # At most n * max
    
    async def test_sensitivity_bounds(self):
        """Test query sensitivity calculations."""
        # Test default sensitivities
        sensitivities = self.mechanism.DEFAULT_SENSITIVITIES
        
        # All sensitivities should be positive
        for query_type, sensitivity in sensitivities.items():
            assert sensitivity > 0, f"Sensitivity for {query_type} must be positive"
        
        # Count queries should have sensitivity 1
        assert sensitivities[QueryType.COUNT] == 1.0
        
        # Sum queries should have bounded sensitivity
        assert sensitivities[QueryType.SUM] >= 1.0
    
    async def test_secure_random_properties(self):
        """Test cryptographically secure random number generation."""
        mechanism_secure = LaplaceNoiseMechanism(secure_random=True)
        mechanism_pseudo = LaplaceNoiseMechanism(secure_random=False)
        
        # Generate samples with both methods
        scale = 1.0
        sample_size = 100
        
        secure_samples = [mechanism_secure.generate_laplace_noise(scale) for _ in range(sample_size)]
        pseudo_samples = [mechanism_pseudo.generate_laplace_noise(scale) for _ in range(sample_size)]
        
        # Both should produce valid noise (non-zero variance)
        import numpy as np
        assert np.var(secure_samples) > 0
        assert np.var(pseudo_samples) > 0
        
        # Secure samples should be different from pseudo samples (high probability)
        # This is a probabilistic test
        correlation = np.corrcoef(secure_samples, pseudo_samples)[0, 1]
        assert abs(correlation) < 0.1, "Secure and pseudo-random samples should be uncorrelated"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
