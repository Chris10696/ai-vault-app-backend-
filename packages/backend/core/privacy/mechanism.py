"""
Laplace mechanism implementation for differential privacy.
Provides secure noise generation and privacy parameter calculation.
"""

import numpy as np
import logging
from typing import Union, List, Optional, Tuple
from scipy import stats
import hashlib
import secrets
from datetime import datetime

from .models import PrivacyParameters, QueryType, PrivacyLevel

logger = logging.getLogger(__name__)

class LaplaceNoiseMechanism:
    """
    Implementation of the Laplace mechanism for differential privacy.
    Adds calibrated Laplace noise to query results to ensure ε-differential privacy.
    """
    
    # Privacy level to epsilon mapping
    PRIVACY_LEVELS = {
        PrivacyLevel.MINIMAL: 1.0,
        PrivacyLevel.STANDARD: 0.5,
        PrivacyLevel.HIGH: 0.1,
        PrivacyLevel.MAXIMUM: 0.01
    }
    
    # Query sensitivity defaults (can be overridden)
    DEFAULT_SENSITIVITIES = {
        QueryType.COUNT: 1.0,
        QueryType.SUM: 1.0,     # Assumes normalized data
        QueryType.MEAN: 1.0,    # Sensitivity depends on data range
        QueryType.MEDIAN: 1.0,
        QueryType.HISTOGRAM: 1.0,
        QueryType.RANGE_QUERY: 1.0
    }
    
    def __init__(self, secure_random: bool = True):
        """
        Initialize the Laplace mechanism.
        
        Args:
            secure_random: Use cryptographically secure random number generation
        """
        self.secure_random = secure_random
        logger.info("LaplaceNoiseMechanism initialized")
    
    def calculate_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """
        Calculate the scale parameter for Laplace noise.
        
        Args:
            sensitivity: Query sensitivity (Δf)
            epsilon: Privacy parameter (ε)
            
        Returns:
            Scale parameter b = Δf/ε
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
            
        scale = sensitivity / epsilon
        logger.debug(f"Calculated noise scale: {scale} (sensitivity={sensitivity}, epsilon={epsilon})")
        return scale
    
    def generate_laplace_noise(self, scale: float, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Generate Laplace noise with specified scale parameter.
        
        Args:
            scale: Scale parameter (b) for Laplace distribution
            size: Number of noise samples to generate (None for single value)
            
        Returns:
            Laplace noise sample(s)
        """
        if self.secure_random:
            # Use cryptographically secure random generation
            if size is None:
                # Generate single secure random value
                uniform_random = secrets.randbits(64) / (2**64)  # [0,1)
                # Transform to (-0.5, 0.5) range
                uniform_random = uniform_random - 0.5
                # Apply inverse CDF of Laplace distribution
                if uniform_random >= 0:
                    noise = -scale * np.log(1 - 2 * uniform_random)
                else:
                    noise = scale * np.log(1 + 2 * uniform_random)
                return noise
            else:
                # Generate multiple secure random values
                noise_samples = []
                for _ in range(size):
                    uniform_random = secrets.randbits(64) / (2**64)
                    uniform_random = uniform_random - 0.5
                    if uniform_random >= 0:
                        noise = -scale * np.log(1 - 2 * uniform_random)
                    else:
                        noise = scale * np.log(1 + 2 * uniform_random)
                    noise_samples.append(noise)
                return np.array(noise_samples)
        else:
            # Use NumPy's pseudo-random generation (faster but less secure)
            return np.random.laplace(loc=0, scale=scale, size=size)
    
    def add_noise(
        self, 
        true_result: Union[float, int, List[Union[float, int]]], 
        privacy_params: PrivacyParameters
    ) -> Tuple[Union[float, List[float]], float]:
        """
        Add Laplace noise to query result.
        
        Args:
            true_result: True query result
            privacy_params: Privacy parameters (epsilon, delta, sensitivity)
            
        Returns:
            Tuple of (noisy_result, actual_noise_added)
        """
        scale = self.calculate_noise_scale(privacy_params.sensitivity, privacy_params.epsilon)
        
        if isinstance(true_result, (list, np.ndarray)):
            # Handle vector results (e.g., histograms)
            noise_vector = self.generate_laplace_noise(scale, size=len(true_result))
            noisy_result = [float(val + noise) for val, noise in zip(true_result, noise_vector)]
            total_noise = float(np.linalg.norm(noise_vector))  # L2 norm of noise vector
            
            logger.debug(f"Added vector noise: scale={scale}, total_noise={total_noise}")
            return noisy_result, total_noise
        else:
            # Handle scalar results
            noise = self.generate_laplace_noise(scale)
            noisy_result = float(true_result + noise)
            
            logger.debug(f"Added scalar noise: {noise} (scale={scale})")
            return noisy_result, abs(float(noise))
    
    def get_privacy_parameters(
        self, 
        query_type: QueryType, 
        privacy_level: PrivacyLevel,
        custom_epsilon: Optional[float] = None,
        custom_sensitivity: Optional[float] = None
    ) -> PrivacyParameters:
        """
        Get privacy parameters for a specific query type and privacy level.
        
        Args:
            query_type: Type of query
            privacy_level: Desired privacy level
            custom_epsilon: Custom epsilon value (overrides privacy level)
            custom_sensitivity: Custom sensitivity (overrides default)
            
        Returns:
            PrivacyParameters object
        """
        epsilon = custom_epsilon if custom_epsilon is not None else self.PRIVACY_LEVELS[privacy_level]
        sensitivity = custom_sensitivity if custom_sensitivity is not None else self.DEFAULT_SENSITIVITIES[query_type]
        
        return PrivacyParameters(
            epsilon=epsilon,
            sensitivity=sensitivity,
            delta=1e-5  # Standard delta for pure DP
        )
    
    def estimate_accuracy_loss(self, privacy_params: PrivacyParameters, confidence: float = 0.95) -> float:
        """
        Estimate expected accuracy loss due to noise addition.
        
        Args:
            privacy_params: Privacy parameters
            confidence: Confidence level for accuracy bound
            
        Returns:
            Expected accuracy loss (standard deviation of noise)
        """
        scale = self.calculate_noise_scale(privacy_params.sensitivity, privacy_params.epsilon)
        
        # For Laplace distribution, standard deviation = sqrt(2) * scale
        std_dev = np.sqrt(2) * scale
        
        # For given confidence level, calculate the bound
        # For Laplace distribution, P(|X| <= t) = 1 - exp(-t/scale)
        # Solving for t: t = -scale * ln(1 - confidence)
        confidence_bound = -scale * np.log(1 - confidence)
        
        logger.debug(f"Accuracy estimates: std_dev={std_dev}, {confidence*100}% bound={confidence_bound}")
        return confidence_bound
    
    def validate_privacy_budget(self, requested_epsilon: float, available_budget: float) -> bool:
        """
        Validate if requested epsilon is within available privacy budget.
        
        Args:
            requested_epsilon: Requested privacy parameter
            available_budget: Available privacy budget
            
        Returns:
            True if request is valid
        """
        return requested_epsilon <= available_budget
    
    def compose_privacy_loss(self, epsilon_values: List[float]) -> float:
        """
        Calculate composed privacy loss using sequential composition.
        
        Args:
            epsilon_values: List of epsilon values from multiple queries
            
        Returns:
            Total composed epsilon
        """
        # For sequential composition, total epsilon is sum of individual epsilons
        total_epsilon = sum(epsilon_values)
        
        logger.debug(f"Composed privacy loss: {epsilon_values} -> {total_epsilon}")
        return total_epsilon
    
    def calculate_optimal_epsilon(
        self, 
        target_accuracy: float, 
        sensitivity: float, 
        confidence: float = 0.95
    ) -> float:
        """
        Calculate optimal epsilon for desired accuracy level.
        
        Args:
            target_accuracy: Desired accuracy bound
            sensitivity: Query sensitivity
            confidence: Confidence level
            
        Returns:
            Recommended epsilon value
        """
        # For Laplace mechanism: accuracy_bound = -sensitivity/epsilon * ln(1 - confidence)
        # Solving for epsilon: epsilon = -sensitivity * ln(1 - confidence) / target_accuracy
        
        if target_accuracy <= 0:
            raise ValueError("Target accuracy must be positive")
        
        epsilon = -sensitivity * np.log(1 - confidence) / target_accuracy
        
        # Clamp to reasonable bounds
        epsilon = max(0.01, min(10.0, epsilon))
        
        logger.debug(f"Calculated optimal epsilon: {epsilon} for accuracy={target_accuracy}")
        return epsilon
