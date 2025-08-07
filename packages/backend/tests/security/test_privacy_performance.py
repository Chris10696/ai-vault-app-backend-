"""
Performance tests for Module B: Differential Privacy Engine.
Tests response times, throughput, and scalability of privacy operations.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ...core.privacy.mechanism import LaplaceNoiseMechanism, PrivacyParameters
from ...core.privacy.vault import FernetKeyManager, EncryptedUserVault, VaultCreateRequest
from ...core.privacy.models import QueryType, PrivacyLevel

class TestPrivacyPerformance:
    """Performance benchmarks for privacy operations."""
    
    def setup_method(self):
        """Set up performance testing environment."""
        self.mechanism = LaplaceNoiseMechanism(secure_random=True)
        self.fast_mechanism = LaplaceNoiseMechanism(secure_random=False)
    
    @pytest.mark.benchmark
    def test_noise_generation_performance(self, benchmark):
        """Benchmark noise generation performance."""
        scale = 1.0
        
        def generate_noise():
            return self.mechanism.generate_laplace_noise(scale)
        
        # Benchmark single noise generation
        result = benchmark(generate_noise)
        assert isinstance(result, float)
    
    @pytest.mark.benchmark
    def test_vector_noise_performance(self, benchmark):
        """Benchmark vector noise generation."""
        scale = 1.0
        vector_size = 1000
        
        def generate_vector_noise():
            return self.mechanism.generate_laplace_noise(scale, size=vector_size)
        
        result = benchmark(generate_vector_noise)
        assert len(result) == vector_size
    
    @pytest.mark.benchmark
    def test_privacy_parameter_calculation(self, benchmark):
        """Benchmark privacy parameter calculation."""
        
        def calculate_params():
            return self.mechanism.get_privacy_parameters(
                QueryType.COUNT,
                PrivacyLevel.STANDARD
            )
        
        result = benchmark(calculate_params)
        assert result.epsilon == 0.5
    
    @pytest.mark.benchmark
    def test_key_derivation_performance(self, benchmark):
        """Benchmark PBKDF2 key derivation."""
        password = "test_password_123"
        salt = FernetKeyManager.generate_salt()
        
        def derive_key():
            return FernetKeyManager.derive_key_from_password(password, salt)
        
        result = benchmark(derive_key)
        assert len(result) == 44  # Base64 encoded 32-byte key
    
    @pytest.mark.benchmark 
    def test_encryption_performance(self, benchmark):
        """Benchmark Fernet encryption performance."""
        key = FernetKeyManager.generate_key()
        vault = EncryptedUserVault("test-user", key)
        
        test_data = {
            "username": "testuser",
            "password": "secretpassword",
            "api_keys": ["key1", "key2", "key3"],
            "metadata": {
                "created": "2024-01-01",
                "updated": "2024-01-15",
                "version": "1.0"
            }
        }
        
        def encrypt_data():
            return vault.encrypt_data(test_data)
        
        result = benchmark(encrypt_data)
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.benchmark
    def test_decryption_performance(self, benchmark):
        """Benchmark Fernet decryption performance."""
        key = FernetKeyManager.generate_key()
        vault = EncryptedUserVault("test-user", key)
        
        test_data = {"test": "data", "numbers": [1, 2, 3, 4, 5]}
        encrypted_data = vault.encrypt_data(test_data)
        
        def decrypt_data():
            return vault.decrypt_data(encrypted_data)
        
        result = benchmark(decrypt_data)
        assert result == test_data
    
    def test_concurrent_noise_generation(self):
        """Test performance under concurrent load."""
        scale = 1.0
        num_threads = 10
        operations_per_thread = 100
        
        def generate_noise_batch():
            """Generate a batch of noise samples."""
            return [self.mechanism.generate_laplace_noise(scale) for _ in range(operations_per_thread)]
        
        start_time = time.time()
        
        # Run concurrent noise generation
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(generate_noise_batch) for _ in range(num_threads)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        total_operations = num_threads * operations_per_thread
        
        # Performance assertions
        assert total_time < 5.0, f"Concurrent noise generation took too long: {total_time}s"
        assert len(results) == num_threads
        assert all(len(batch) == operations_per_thread for batch in results)
        
        # Calculate throughput
        throughput = total_operations / total_time
        print(f"Noise generation throughput: {throughput:.2f} operations/second")
        assert throughput > 100, f"Throughput too low: {throughput} ops/sec"
    
    def test_secure_vs_pseudo_random_performance(self):
        """Compare performance of secure vs pseudo-random generation."""
        scale = 1.0
        iterations = 1000
        
        # Benchmark secure random
        start_time = time.time()
        secure_samples = [self.mechanism.generate_laplace_noise(scale) for _ in range(iterations)]
        secure_time = time.time() - start_time
        
        # Benchmark pseudo-random
        start_time = time.time()
        pseudo_samples = [self.fast_mechanism.generate_laplace_noise(scale) for _ in range(iterations)]
        pseudo_time = time.time() - start_time
        
        # Performance comparison
        print(f"Secure random: {secure_time:.4f}s ({iterations/secure_time:.1f} ops/sec)")
        print(f"Pseudo random: {pseudo_time:.4f}s ({iterations/pseudo_time:.1f} ops/sec)")
        
        # Secure should be slower but not more than 10x
        assert secure_time > pseudo_time, "Secure random should be slower"
        assert secure_time / pseudo_time < 10, "Secure random shouldn't be more than 10x slower"
        
        # Both should produce valid results
        assert len(secure_samples) == iterations
        assert len(pseudo_samples) == iterations
    
    @pytest.mark.asyncio
    async def test_async_operation_performance(self):
        """Test performance of async privacy operations."""
        num_operations = 100
        
        async def async_privacy_operation():
            """Simulate an async privacy operation."""
            # Simulate some async work
            await asyncio.sleep(0.001)  # 1ms simulated I/O
            
            # Perform privacy calculation
            privacy_params = self.mechanism.get_privacy_parameters(
                QueryType.COUNT,
                PrivacyLevel.STANDARD
            )
            noisy_result, _ = self.mechanism.add_noise(100.0, privacy_params)
            return noisy_result
        
        start_time = time.time()
        
        # Run operations concurrently
        tasks = [async_privacy_operation() for _ in range(num_operations)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert len(results) == num_operations
        assert total_time < 2.0, f"Async operations took too long: {total_time}s"
        
        # Calculate throughput
        throughput = num_operations / total_time
        print(f"Async privacy operations throughput: {throughput:.2f} ops/sec")
        assert throughput > 50, f"Async throughput too low: {throughput} ops/sec"
    
    def test_memory_usage_noise_generation(self):
        """Test memory usage during large-scale noise generation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large amount of noise
        scale = 1.0
        large_vector_size = 100000
        
        noise_vector = self.mechanism.generate_laplace_noise(scale, size=large_vector_size)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory usage should be reasonable
        assert len(noise_vector) == large_vector_size
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f} MB"
        
        print(f"Memory usage for {large_vector_size} samples: {memory_increase:.2f} MB")
    
    def test_encryption_throughput(self):
        """Test encryption/decryption throughput for various data sizes."""
        key = FernetKeyManager.generate_key()
        vault = EncryptedUserVault("test-user", key)
        
        # Test different data sizes
        data_sizes = [
            ("small", {"key": "value"}),
            ("medium", {"keys": [f"value_{i}" for i in range(100)]}),
            ("large", {"data": {"nested": {"values": [i for i in range(1000)]}}})
        ]
        
        for size_name, test_data in data_sizes:
            # Measure encryption
            start_time = time.time()
            encrypted = vault.encrypt_data(test_data)
            encryption_time = time.time() - start_time
            
            # Measure decryption
            start_time = time.time()
            decrypted = vault.decrypt_data(encrypted)
            decryption_time = time.time() - start_time
            
            # Verify correctness
            assert decrypted == test_data
            
            # Performance logging
            print(f"{size_name} data - Encryption: {encryption_time*1000:.2f}ms, Decryption: {decryption_time*1000:.2f}ms")
            
            # Performance assertions (adjust based on requirements)
            assert encryption_time < 0.1, f"Encryption too slow for {size_name} data: {encryption_time}s"
            assert decryption_time < 0.1, f"Decryption too slow for {size_name} data: {decryption_time}s"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
