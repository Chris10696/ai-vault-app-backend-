import numpy as np
from typing import Union, List, Optional
import logging
from scipy.stats import laplace
import hashlib

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """
    Differential Privacy implementation for recommendation systems
    
    Provides mechanisms for adding calibrated noise to protect user privacy
    while maintaining data utility for machine learning models.
    """
    
    def __init__(self, epsilon: float = 0.5, delta: float = 1e-5):
        """
        Initialize differential privacy parameters
        
        Args:
            epsilon: Privacy budget parameter (smaller = more private)
            delta: Failure probability for approximate DP
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta < 0 or delta >= 1:
            raise ValueError("Delta must be in [0, 1)")
            
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_spent = 0.0
        
        logger.info(f"Initialized DP with epsilon={epsilon}, delta={delta}")
    
    def add_laplace_noise(
        self,
        data: Union[np.ndarray, List, float],
        sensitivity: float,
        size: Optional[int] = None
    ) -> np.ndarray:
        """
        Add Laplace noise for epsilon-differential privacy
        
        Args:
            data: Input data to add noise to
            sensitivity: Global sensitivity of the query
            size: Size of noise array if data is scalar
            
        Returns:
            Data with added Laplace noise
        """
        if isinstance(data, (int, float)):
            if size is None:
                size = 1
            data_array = np.full(size, data)
        else:
            data_array = np.array(data)
        
        # Calculate noise scale
        scale = sensitivity / self.epsilon
        
        # Generate Laplace noise
        noise = np.random.laplace(0, scale, data_array.shape)
        
        # Update privacy budget
        self.privacy_spent += self.epsilon
        
        logger.debug(f"Added Laplace noise with scale={scale}, privacy_spent={self.privacy_spent}")
        
        return data_array + noise
    
    def add_gaussian_noise(
        self,
        data: Union[np.ndarray, List, float],
        sensitivity: float,
        size: Optional[int] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise for (epsilon, delta)-differential privacy
        
        Args:
            data: Input data to add noise to
            sensitivity: L2 sensitivity of the query
            size: Size of noise array if data is scalar
            
        Returns:
            Data with added Gaussian noise
        """
        if isinstance(data, (int, float)):
            if size is None:
                size = 1
            data_array = np.full(size, data)
        else:
            data_array = np.array(data)
        
        # Calculate noise scale for (epsilon, delta)-DP
        if self.delta == 0:
            raise ValueError("Delta must be > 0 for Gaussian mechanism")
        
        c = np.sqrt(2 * np.log(1.25 / self.delta))
        sigma = c * sensitivity / self.epsilon
        
        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, data_array.shape)
        
        # Update privacy budget
        self.privacy_spent += self.epsilon
        
        logger.debug(f"Added Gaussian noise with sigma={sigma}, privacy_spent={self.privacy_spent}")
        
        return data_array + noise
    
    def privatize_ratings(self, ratings: np.ndarray, rating_scale: tuple = (1, 5)) -> np.ndarray:
        """
        Add privacy-preserving noise to user ratings
        
        Args:
            ratings: Array of user ratings
            rating_scale: Min and max rating values
            
        Returns:
            Privatized ratings clipped to valid range
        """
        min_rating, max_rating = rating_scale
        sensitivity = max_rating - min_rating
        
        # Add Laplace noise
        privatized = self.add_laplace_noise(ratings, sensitivity)
        
        # Clip to valid rating range
        privatized = np.clip(privatized, min_rating, max_rating)
        
        return privatized
    
    def check_privacy_budget(self, required_epsilon: float) -> bool:
        """
        Check if there's enough privacy budget remaining
        
        Args:
            required_epsilon: Required epsilon for the operation
            
        Returns:
            True if budget is available
        """
        return (self.privacy_spent + required_epsilon) <= (self.epsilon * 10)  # Allow 10x budget
    
    def reset_privacy_budget(self):
        """Reset the privacy budget counter"""
        self.privacy_spent = 0.0
        logger.info("Privacy budget reset")
    
    def get_privacy_report(self) -> dict:
        """
        Generate a privacy expenditure report
        
        Returns:
            Dictionary with privacy statistics
        """
        return {
            "total_epsilon": self.epsilon,
            "delta": self.delta,
            "epsilon_spent": self.privacy_spent,
            "epsilon_remaining": max(0, (self.epsilon * 10) - self.privacy_spent),
            "privacy_level": "High" if self.epsilon < 1.0 else "Medium" if self.epsilon < 3.0 else "Low"
        }
