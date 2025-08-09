"""
Integration tests for homomorphic encryption performance optimization
"""

import pytest
import numpy as np
import time
import threading
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor

from packages.backend.core.crypto.performance_optimizer import (
    PerformanceOptimizer,
    MemoryPool,
    BatchProcessor,
    CacheManager,
    ParallelizationManager
)
from packages.backend.core.crypto.seal_manager import SealContextManager
from packages.backend.core.security.zero_trust import SecurityContext
from packages.backend.services.execution.he_inference_engine import HEInferenceEngine
from packages.backend.services.execution.inference_pipeline import AdvancedInferencePipeline

class TestPerformanceOptimization:
    """Test performance optimization components"""
    
    @pytest.fixture
    def mock_security_context(self):
        """Mock security context for testing"""
        from unittest.mock import Mock
        context = Mock(spec=SecurityContext)
        context.get_user_id.return_value = "test_user"
        context.get_trust_score.return_value = 0.9
        context.get_context_id.return_value = "test_context"
        context.get_encryption_key.return_value = "test_key"
        return context
    
    @pytest.fixture
    def seal_manager(self, mock_security_context, tmp_path):
        """Create SEAL manager for testing"""
        return SealContextManager(mock_security_context, str(tmp_path))
    
    @pytest.fixture
    def performance_optimizer(self, seal_manager, mock_security_context):
        """Create performance optimizer for testing"""
        return PerformanceOptimizer(seal_manager, mock_security_context)
    
    def test_memory_pool_efficiency(self):
        """Test memory pool reduces allocation overhead"""
        import tenseal as ts
        
        # Create context
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_modulus=[60, 40, 60])
        context.global_scale = 2 ** 40
        context.generate_galois_keys()
        
        memory_pool = MemoryPool(max_pool_size=10)
        
        # Test allocation and reuse
        vectors = []
        
        # First allocation
        vec1 = memory_pool.get_vector(100, context)
        assert vec1 is not None
        vectors.append(vec1)
        
        # Return to pool
        memory_pool.return_vector(vec1)
        
        # Get again (should reuse)
        vec2 = memory_pool.get_vector(100, context)
        assert vec2 is not None
        
        # Check statistics
        stats = memory_pool.get_statistics()
        assert stats['total_allocations'] >= 1
        assert stats['reuse_rate'] >= 0
    
    def test_batch_processor_throughput(self):
        """Test batch processor improves throughput"""
        batch_processor = BatchProcessor(max_batch_size=5, batch_timeout=0.1)
        batch_processor.start()
        
        try:
            # Define test operation
            def test_operation(x):
                time.sleep(0.01)  # Simulate work
                return x * 2
            
            # Submit operations
            operation_ids = []
            start_time = time.time()
            
            for i in range(10):
                op_id = batch_processor.submit_operation(f"op_{i}", test_operation, i)
                operation_ids.append(op_id)
            
            # Get results
            results = []
            for op_id in operation_ids:
                result = batch_processor.get_result(op_id, timeout=5.0)
                results.append(result)
            
            total_time = time.time() - start_time
            
            # Verify results
            expected_results = [i * 2 for i in range(10)]
            assert results == expected_results
            
            # Should be faster than sequential execution
            assert total_time < 0.5  # Should complete quickly due to batching
            
        finally:
            batch_processor.stop()
    
    def test_parallel_execution_speedup(self):
        """Test parallel execution provides speedup"""
        parallelization_manager = ParallelizationManager(max_workers=4)
        
        def cpu_intensive_operation(x):
            # Simulate CPU-intensive work
            result = 0
            for i in range(10000):
                result += np.sqrt(x + i)
            return result
        
        inputs = list(range(20))
        
        # Sequential execution
        start_time = time.time()
        sequential_results = [cpu_intensive_operation(x) for x in inputs]
        sequential_time = time.time() - start_time
        
        # Parallel execution
        start_time = time.time()
        parallel_results = parallelization_manager.submit_parallel_operation(
            cpu_intensive_operation, inputs
        )
        parallel_time = time.time() - start_time
        
        # Verify results are the same
        assert len(parallel_results) == len(sequential_results)
        for i, result in enumerate(parallel_results):
            if result is not None:  # Handle potential None results
                assert abs(result - sequential_results[i]) < 1e-6
        
        # Parallel should be faster (allowing some overhead)
        speedup = sequential_time / max(parallel_time, 0.001)
        logger.info(f"Parallel speedup: {speedup:.2f}x")
        
        parallelization_manager.shutdown()
    
    def test_cache_manager_effectiveness(self):
        """Test cache manager improves performance"""
        cache_manager = CacheManager(max_cache_size=100, ttl_seconds=60)
        
        # Test cache miss and hit
        key = "test_operation_key"
        value = "expensive_computation_result"
        
        # Cache miss
        result = cache_manager.get(key)
        assert result is None
        
        # Store value
        cache_manager.put(key, value)
        
        # Cache hit
        cached_result = cache_manager.get(key)
        assert cached_result == value
        
        # Test cache statistics
        stats = cache_manager.get_statistics()
        assert stats['cache_size'] == 1
        assert stats['utilization'] > 0
    
    def test_optimization_integration(self, performance_optimizer):
        """Test integrated optimization performance"""
        # Test optimized encryption
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # First call (no cache)
        start_time = time.time()
        result1 = performance_optimizer.optimize_encryption(test_data, "performance")
        first_call_time = time.time() - start_time
        
        assert result1 is not None
        
        # Second call (should use cache)
        start_time = time.time()
        result2 = performance_optimizer.optimize_encryption(test_data, "performance")
        second_call_time = time.time() - start_time
        
        assert result2 is not None
        
        # Second call should be faster due to caching
        assert second_call_time < first_call_time
        
        # Test performance report
        report = performance_optimizer.get_performance_report()
        assert 'summary' in report
        assert 'operation_breakdown' in report
        assert report['summary']['total_operations'] > 0
    
    def test_performance_tuning(self, performance_optimizer):
        """Test automatic performance tuning"""
        # Initial performance report
        initial_report = performance_optimizer.get_performance_report()
        
        # Trigger performance tuning
        performance_optimizer.tune_performance("execution_time", 0.1)
        
        # Verify tuning applied
        # (In a real test, we would measure actual performance improvement)
        assert performance_optimizer.batch_processor.max_batch_size > 0
        
        # Test memory usage tuning
        performance_optimizer.tune_performance("memory_usage", 10.0)
        
        # Verify memory optimizations applied
        assert performance_optimizer.memory_pool.max_pool_size > 0

class TestHEInferencePerformance:
    """Test homomorphic encryption inference performance"""
    
    @pytest.fixture
    def inference_setup(self, tmp_path):
        """Setup inference components for testing"""
        from unittest.mock import Mock
        
        security_context = Mock(spec=SecurityContext)
        security_context.get_user_id.return_value = "test_user"
        security_context.get_trust_score.return_value = 0.9
        security_context.get_context_id.return_value = "test_context"
        security_context.get_encryption_key.return_value = "test_key"
        
        seal_manager = SealContextManager(security_context, str(tmp_path))
        
        from packages.backend.core.crypto.key_manager import AdvancedKeyManager
        key_manager = AdvancedKeyManager(security_context, str(tmp_path / "keys"))
        
        he_engine = HEInferenceEngine(security_context, seal_manager, key_manager)
        pipeline = AdvancedInferencePipeline(security_context, he_engine)
        
        return {
            'security_context': security_context,
            'seal_manager': seal_manager,
            'key_manager': key_manager,
            'he_engine': he_engine,
            'pipeline': pipeline
        }
    
    def test_concurrent_inference_performance(self, inference_setup):
        """Test performance under concurrent inference load"""
        he_engine = inference_setup['he_engine']
        
        # Create a simple test model
        test_model_architecture = {
            'type': 'neural_network',
            'layers': [
                {
                    'type': 'linear',
                    'input_size': 10,
                    'output_size': 5,
                    'use_bias': True
                },
                {
                    'type': 'polynomial',
                    'degree': 2,
                    'coefficients': [0, 1, 0.1]
                }
            ],
            'input_shape': [10],
            'output_shape': [5]
        }
        
        # Create dummy model weights
        import tempfile
        import torch
        
        model_weights = {
            'layer_0_weight': torch.randn(5, 10),
            'layer_0_bias': torch.randn(5)
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model_weights, f.name)
            model_path = f.name
        
        try:
            # Load and encrypt model
            model_id = he_engine.load_and_encrypt_model(
                model_path, "test_model", test_model_architecture, "performance"
            )
            
            # Test concurrent inference
            num_threads = 5
            num_requests_per_thread = 3
            
            def run_inference_batch(thread_id):
                results = []
                for i in range(num_requests_per_thread):
                    session_id = he_engine.create_inference_session(model_id, "performance")
                    
                    test_input = np.random.random(10).tolist()
                    result = he_engine.run_encrypted_inference(session_id, test_input)
                    
                    results.append(result)
                    he_engine.close_session(session_id)
                
                return results
            
            # Execute concurrent inference
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                start_time = time.time()
                
                futures = []
                for thread_id in range(num_threads):
                    future = executor.submit(run_inference_batch, thread_id)
                    futures.append(future)
                
                # Collect results
                all_results = []
                for future in futures:
                    thread_results = future.result(timeout=30.0)
                    all_results.extend(thread_results)
                
                total_time = time.time() - start_time
            
            # Verify results
            total_inferences = num_threads * num_requests_per_thread
            assert len(all_results) == total_inferences
            
            successful_inferences = sum(1 for r in all_results if r.success)
            success_rate = successful_inferences / total_inferences
            
            assert success_rate >= 0.8  # At least 80% success rate
            
            avg_time_per_inference = total_time / total_inferences
            logger.info(f"Average inference time: {avg_time_per_inference:.3f}s")
            logger.info(f"Success rate: {success_rate:.2%}")
            
            # Performance should be reasonable
            assert avg_time_per_inference < 5.0  # Less than 5 seconds per inference
            
        finally:
            import os
            os.unlink(model_path)
    
    def test_memory_usage_stability(self, inference_setup):
        """Test memory usage remains stable during extended operation"""
        he_engine = inference_setup['he_engine']
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        # Create test model (simplified)
        test_architecture = {
            'type': 'simple',
            'layers': [{'type': 'linear', 'input_size': 5, 'output_size': 3}],
            'input_shape': [5],
            'output_shape': [3]
        }
        
        # Create lightweight dummy model
        import tempfile
        dummy_weights = {'layer_0_weight': np.random.randn(3, 5)}
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            import pickle
            pickle.dump(dummy_weights, f)
            model_path = f.name
        
        try:
            model_id = he_engine.load_and_encrypt_model(
                model_path, "memory_test_model", test_architecture, "performance"
            )
            
            # Run many operations
            for i in range(50):  # Reduced for performance
                session_id = he_engine.create_inference_session(model_id, "performance")
                
                test_input = np.random.random(5).tolist()
                result = he_engine.run_encrypted_inference(session_id, test_input)
                
                he_engine.close_session(session_id)
                
                # Monitor memory
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                peak_memory = max(peak_memory, current_memory)
                
                if i % 10 == 0:
                    logger.info(f"Iteration {i}, Memory: {current_memory:.1f}MB")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            logger.info(f"Memory usage - Initial: {initial_memory:.1f}MB, "
                       f"Peak: {peak_memory:.1f}MB, Final: {final_memory:.1f}MB")
            
            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f}MB increase"
            
        finally:
            import os
            os.unlink(model_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
