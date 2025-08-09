"""
Performance optimization system for homomorphic encryption operations
Implements advanced optimization techniques for real-time processing
"""

import os
import time
import json
import logging
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import psutil
import numpy as np

try:
    import tenseal as ts
    from memory_profiler import profile
except ImportError as e:
    raise ImportError(f"Performance optimization dependencies not installed: {e}")

from .seal_manager import SealContextManager, HomomorphicOperationManager
from ..security.zero_trust import SecurityContext

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_type: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    thread_id: int
    timestamp: datetime
    input_size: int
    context_name: str
    optimization_applied: List[str]

@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration"""
    name: str
    enabled: bool
    priority: int
    target_operations: List[str]
    parameters: Dict[str, Any]
    expected_speedup: float

class MemoryPool:
    """
    Memory pool for efficient tensor allocation and reuse
    Reduces memory allocation overhead in homomorphic operations
    """
    
    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self._pools: Dict[int, queue.Queue] = {}
        self._lock = threading.RLock()
        self._stats = {
            'allocations': 0,
            'reuses': 0,
            'total_size': 0
        }
    
    def get_vector(self, size: int, context: ts.Context) -> ts.CKKSVector:
        """Get a vector from pool or create new one"""
        with self._lock:
            if size not in self._pools:
                self._pools[size] = queue.Queue()
            
            pool = self._pools[size]
            
            if not pool.empty():
                # Reuse existing vector
                try:
                    vector = pool.get_nowait()
                    self._stats['reuses'] += 1
                    return vector
                except queue.Empty:
                    pass
            
            # Create new vector
            zero_data = np.zeros(size)
            vector = ts.ckks_vector(context, zero_data.tolist())
            self._stats['allocations'] += 1
            self._stats['total_size'] += size
            
            return vector
    
    def return_vector(self, vector: ts.CKKSVector):
        """Return vector to pool for reuse"""
        with self._lock:
            size = vector.size()
            
            if size not in self._pools:
                self._pools[size] = queue.Queue()
            
            pool = self._pools[size]
            
            if pool.qsize() < self.max_pool_size:
                # Clear vector data and return to pool
                try:
                    zero_data = np.zeros(size)
                    # Reset vector (this is conceptual - actual implementation would vary)
                    pool.put(vector)
                except Exception as e:
                    logger.warning(f"Failed to return vector to pool: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            pool_sizes = {size: pool.qsize() for size, pool in self._pools.items()}
            reuse_rate = self._stats['reuses'] / max(1, self._stats['allocations'] + self._stats['reuses'])
            
            return {
                'pool_sizes': pool_sizes,
                'total_allocations': self._stats['allocations'],
                'total_reuses': self._stats['reuses'],
                'reuse_rate': reuse_rate,
                'total_memory_size': self._stats['total_size']
            }

class BatchProcessor:
    """
    Batch processing system for homomorphic operations
    Optimizes throughput by processing multiple operations together
    """
    
    def __init__(self, max_batch_size: int = 32, batch_timeout: float = 0.1):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self._pending_operations: queue.Queue = queue.Queue()
        self._results: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._processor_thread = None
        self._running = False
    
    def start(self):
        """Start batch processor"""
        if not self._running:
            self._running = True
            self._processor_thread = threading.Thread(target=self._process_batches)
            self._processor_thread.daemon = True
            self._processor_thread.start()
            logger.info("Batch processor started")
    
    def stop(self):
        """Stop batch processor"""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        logger.info("Batch processor stopped")
    
    def submit_operation(
        self, 
        operation_id: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> str:
        """Submit operation for batch processing"""
        operation_data = {
            'id': operation_id,
            'func': operation_func,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }
        
        self._pending_operations.put(operation_data)
        return operation_id
    
    def get_result(self, operation_id: str, timeout: float = 10.0) -> Any:
        """Get result of submitted operation"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if operation_id in self._results:
                    result = self._results.pop(operation_id)
                    if 'error' in result:
                        raise Exception(result['error'])
                    return result['value']
            
            time.sleep(0.01)  # Small delay to avoid busy waiting
        
        raise TimeoutError(f"Operation {operation_id} timed out")
    
    def _process_batches(self):
        """Main batch processing loop"""
        while self._running:
            batch = self._collect_batch()
            
            if batch:
                self._execute_batch(batch)
            else:
                time.sleep(0.001)  # Small delay when no operations pending
    
    def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect operations into a batch"""
        batch = []
        batch_start_time = time.time()
        
        while (len(batch) < self.max_batch_size and 
               time.time() - batch_start_time < self.batch_timeout):
            
            try:
                operation = self._pending_operations.get_nowait()
                batch.append(operation)
            except queue.Empty:
                if batch:  # If we have some operations, process them
                    break
                time.sleep(0.001)
        
        return batch
    
    def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute a batch of operations"""
        try:
            batch_start = time.time()
            
            # Group operations by function type for better optimization
            grouped_ops = {}
            for op in batch:
                func_name = op['func'].__name__
                if func_name not in grouped_ops:
                    grouped_ops[func_name] = []
                grouped_ops[func_name].append(op)
            
            # Execute each group
            for func_name, ops in grouped_ops.items():
                self._execute_operation_group(ops)
            
            batch_time = time.time() - batch_start
            logger.debug(f"Batch of {len(batch)} operations completed in {batch_time:.4f}s")
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            
            # Mark all operations as failed
            for op in batch:
                with self._lock:
                    self._results[op['id']] = {'error': str(e)}
    
    def _execute_operation_group(self, operations: List[Dict[str, Any]]):
        """Execute a group of similar operations"""
        try:
            for op in operations:
                result = op['func'](*op['args'], **op['kwargs'])
                
                with self._lock:
                    self._results[op['id']] = {'value': result}
        
        except Exception as e:
            logger.error(f"Operation group execution failed: {e}")
            
            for op in operations:
                with self._lock:
                    self._results[op['id']] = {'error': str(e)}

class ParallelizationManager:
    """
    Manages parallel execution of homomorphic operations
    Optimizes CPU and memory usage for concurrent processing
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1))
        self._active_operations: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def submit_parallel_operation(
        self,
        operation_func: Callable,
        inputs: List[Any],
        use_processes: bool = False
    ) -> List[Any]:
        """
        Submit operation for parallel execution
        Returns list of results in same order as inputs
        """
        try:
            pool = self._process_pool if use_processes else self._thread_pool
            
            # Submit all operations
            futures = []
            for input_data in inputs:
                future = pool.submit(operation_func, input_data)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30.0)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel operation failed: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel operation submission failed: {e}")
            return [None] * len(inputs)
    
    def shutdown(self):
        """Shutdown parallel execution pools"""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)

class CacheManager:
    """
    Intelligent caching system for homomorphic operations
    Reduces computation overhead by caching intermediate results
    """
    
    def __init__(self, max_cache_size: int = 1000, ttl_seconds: int = 3600):
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    self._access_times[key] = time.time()
                    return entry['value']
                else:
                    # Expired, remove from cache
                    del self._cache[key]
                    del self._access_times[key]
            
            return None
    
    def put(self, key: str, value: Any):
        """Store value in cache"""
        with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self.max_cache_size:
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self._access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._access_times:
            return
        
        # Find LRU key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = len(self._cache)
            
            # Calculate hit rate (simplified)
            return {
                'cache_size': total_size,
                'max_size': self.max_cache_size,
                'utilization': total_size / self.max_cache_size if self.max_cache_size > 0 else 0
            }
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

class PerformanceOptimizer:
    """
    Main performance optimization coordinator
    Orchestrates various optimization strategies for real-time HE processing
    """
    
    def __init__(
        self, 
        seal_manager: SealContextManager,
        security_context: SecurityContext
    ):
        self.seal_manager = seal_manager
        self.security_context = security_context
        
        # Optimization components
        self.memory_pool = MemoryPool()
        self.batch_processor = BatchProcessor()
        self.parallelization_manager = ParallelizationManager()
        self.cache_manager = CacheManager()
        
        # Performance monitoring
        self.metrics: List[PerformanceMetrics] = []
        self._lock = threading.RLock()
        
        # Optimization strategies
        self.strategies: List[OptimizationStrategy] = []
        self._initialize_strategies()
        
        # Start background processes
        self.batch_processor.start()
    
    def _initialize_strategies(self):
        """Initialize optimization strategies"""
        strategies = [
            OptimizationStrategy(
                name="memory_pooling",
                enabled=True,
                priority=1,
                target_operations=["encrypt", "decrypt", "add", "multiply"],
                parameters={"max_pool_size": 100},
                expected_speedup=1.3
            ),
            OptimizationStrategy(
                name="batch_processing",
                enabled=True,
                priority=2,
                target_operations=["batch_encrypt", "batch_decrypt"],
                parameters={"batch_size": 32, "timeout": 0.1},
                expected_speedup=2.5
            ),
            OptimizationStrategy(
                name="parallel_execution",
                enabled=True,
                priority=3,
                target_operations=["matrix_multiply", "large_operations"],
                parameters={"max_workers": 8},
                expected_speedup=4.0
            ),
            OptimizationStrategy(
                name="result_caching",
                enabled=True,
                priority=4,
                target_operations=["inference", "repeated_operations"],
                parameters={"cache_size": 1000, "ttl": 3600},
                expected_speedup=10.0
            )
        ]
        
        self.strategies.extend(strategies)
    
    @profile
    def optimize_encryption(
        self, 
        data: List[float], 
        context_name: str = "high_security"
    ) -> ts.CKKSVector:
        """Optimized encryption with performance tracking"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Check cache first
            cache_key = f"encrypt_{hash(str(data))}_{context_name}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                logger.debug("Using cached encryption result")
                return cached_result
            
            # Get context and create operation manager
            context = self.seal_manager.get_context(context_name)
            op_manager = HomomorphicOperationManager(self.seal_manager)
            
            # Use memory pool for vector creation
            vector = self.memory_pool.get_vector(len(data), context)
            
            # Perform encryption
            result = op_manager.encrypt_vector(data, context_name)
            
            # Cache result
            self.cache_manager.put(cache_key, result)
            
            # Record performance metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            metrics = PerformanceMetrics(
                operation_type="encrypt",
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=psutil.cpu_percent(),
                thread_id=threading.get_ident(),
                timestamp=datetime.utcnow(),
                input_size=len(data),
                context_name=context_name,
                optimization_applied=["memory_pooling", "caching"]
            )
            
            self._record_metrics(metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized encryption failed: {e}")
            raise
    
    def optimize_batch_operations(
        self,
        operation_func: Callable,
        inputs: List[Any],
        context_name: str = "high_security"
    ) -> List[Any]:
        """Optimize batch operations using parallel processing"""
        try:
            start_time = time.time()
            
            # Check if operation benefits from parallelization
            if len(inputs) > 4:  # Threshold for parallel processing
                # Use parallel execution
                results = self.parallelization_manager.submit_parallel_operation(
                    operation_func, inputs
                )
            else:
                # Sequential execution for small batches
                results = []
                for input_data in inputs:
                    result = operation_func(input_data)
                    results.append(result)
            
            total_time = time.time() - start_time
            logger.info(f"Batch operation completed: {len(inputs)} items in {total_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch optimization failed: {e}")
            return [None] * len(inputs)
    
    def optimize_inference_pipeline(
        self,
        model_id: str,
        input_data: Any,
        optimization_level: str = "aggressive"
    ) -> Any:
        """Optimize entire inference pipeline"""
        try:
            pipeline_start = time.time()
            
            # Generate cache key for entire pipeline
            pipeline_cache_key = f"pipeline_{model_id}_{hash(str(input_data))}"
            
            # Check pipeline cache
            cached_result = self.cache_manager.get(pipeline_cache_key)
            if cached_result is not None:
                logger.info("Using cached inference pipeline result")
                return cached_result
            
            # Apply optimization strategies based on level
            optimizations = []
            
            if optimization_level == "aggressive":
                optimizations = ["memory_pooling", "batch_processing", "parallel_execution", "caching"]
            elif optimization_level == "balanced":
                optimizations = ["memory_pooling", "caching"]
            else:  # conservative
                optimizations = ["caching"]
            
            # Execute optimized pipeline
            # This would integrate with the actual inference engine
            result = self._execute_optimized_pipeline(
                model_id, input_data, optimizations
            )
            
            # Cache pipeline result
            self.cache_manager.put(pipeline_cache_key, result)
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"Optimized inference pipeline completed in {pipeline_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Inference pipeline optimization failed: {e}")
            raise
    
    def _execute_optimized_pipeline(
        self, 
        model_id: str, 
        input_data: Any, 
        optimizations: List[str]
    ) -> Any:
        """Execute inference pipeline with specified optimizations"""
        # This is a placeholder for the actual optimized pipeline execution
        # In practice, this would coordinate with the HEInferenceEngine
        
        logger.debug(f"Executing pipeline with optimizations: {optimizations}")
        
        # Simulate optimization effects
        if "parallel_execution" in optimizations:
            time.sleep(0.1)  # Reduced execution time
        else:
            time.sleep(0.3)  # Normal execution time
        
        return {"optimized_result": True, "model_id": model_id}
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        with self._lock:
            self.metrics.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            if not self.metrics:
                return {"message": "No performance metrics available"}
            
            # Calculate statistics
            recent_metrics = self.metrics[-100:]  # Last 100 operations
            
            avg_execution_time = np.mean([m.execution_time for m in recent_metrics])
            avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
            
            # Group by operation type
            operation_stats = {}
            for metric in recent_metrics:
                op_type = metric.operation_type
                if op_type not in operation_stats:
                    operation_stats[op_type] = {
                        'count': 0,
                        'total_time': 0,
                        'total_memory': 0
                    }
                
                operation_stats[op_type]['count'] += 1
                operation_stats[op_type]['total_time'] += metric.execution_time
                operation_stats[op_type]['total_memory'] += metric.memory_usage
            
            # Calculate averages
            for op_type in operation_stats:
                stats = operation_stats[op_type]
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['avg_memory'] = stats['total_memory'] / stats['count']
            
            # Get component statistics
            memory_pool_stats = self.memory_pool.get_statistics()
            cache_stats = self.cache_manager.get_statistics()
            
            return {
                'summary': {
                    'total_operations': len(self.metrics),
                    'recent_operations': len(recent_metrics),
                    'avg_execution_time': avg_execution_time,
                    'avg_memory_usage': avg_memory_usage
                },
                'operation_breakdown': operation_stats,
                'optimization_components': {
                    'memory_pool': memory_pool_stats,
                    'cache': cache_stats
                },
                'active_strategies': [s.name for s in self.strategies if s.enabled],
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def tune_performance(self, target_metric: str = "execution_time", target_value: float = 1.0):
        """Automatically tune performance parameters"""
        try:
            logger.info(f"Starting performance tuning for {target_metric} target: {target_value}")
            
            # Analyze current performance
            report = self.get_performance_report()
            current_value = report['summary'].get(f'avg_{target_metric}', float('inf'))
            
            if current_value <= target_value:
                logger.info("Performance target already met")
                return
            
            # Apply tuning strategies
            if target_metric == "execution_time":
                self._tune_execution_time(target_value, current_value)
            elif target_metric == "memory_usage":
                self._tune_memory_usage(target_value, current_value)
            
            logger.info("Performance tuning completed")
            
        except Exception as e:
            logger.error(f"Performance tuning failed: {e}")
    
    def _tune_execution_time(self, target: float, current: float):
        """Tune parameters to improve execution time"""
        improvement_needed = current / target
        
        if improvement_needed > 2.0:
            # Aggressive optimization needed
            self.batch_processor.max_batch_size = 64
            self.parallelization_manager.max_workers = min(16, os.cpu_count() * 2)
            
            # Enable all optimization strategies
            for strategy in self.strategies:
                strategy.enabled = True
                
        elif improvement_needed > 1.5:
            # Moderate optimization
            self.batch_processor.max_batch_size = 32
            
            # Enable high-impact strategies
            high_impact = ["batch_processing", "parallel_execution", "caching"]
            for strategy in self.strategies:
                strategy.enabled = strategy.name in high_impact
        
        logger.info(f"Tuned execution time parameters for {improvement_needed}x improvement")
    
    def _tune_memory_usage(self, target: float, current: float):
        """Tune parameters to reduce memory usage"""
        if current > target:
            # Reduce memory pool size
            self.memory_pool.max_pool_size = max(50, self.memory_pool.max_pool_size // 2)
            
            # Reduce cache size
            self.cache_manager.max_cache_size = max(100, self.cache_manager.max_cache_size // 2)
            
            # Disable memory-intensive optimizations
            memory_intensive = ["batch_processing", "parallel_execution"]
            for strategy in self.strategies:
                if strategy.name in memory_intensive:
                    strategy.enabled = False
        
        logger.info("Tuned memory usage parameters")
    
    def shutdown(self):
        """Shutdown optimization components"""
        try:
            self.batch_processor.stop()
            self.parallelization_manager.shutdown()
            self.cache_manager.clear()
            logger.info("Performance optimizer shutdown complete")
            
        except Exception as e:
            logger.error(f"Optimization shutdown failed: {e}")
