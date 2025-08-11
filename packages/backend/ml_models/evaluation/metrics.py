import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import math

logger = logging.getLogger(__name__)

class RecommendationMetrics:
    """
    Comprehensive evaluation metrics for recommendation systems
    """
    
    def __init__(self):
        pass
    
    def precision_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if not recommended or k <= 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / min(len(recommended_k), k)
    
    def recall_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if not relevant or k <= 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / len(relevant_set)
    
    def f1_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate F1@K
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: Number of top recommendations to consider
            
        Returns:
            F1@K score
        """
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int,
        relevance_scores: Dict[str, float] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)@K
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items  
            k: Number of top recommendations to consider
            relevance_scores: Dictionary mapping items to relevance scores
            
        Returns:
            NDCG@K score
        """
        if not recommended or not relevant or k <= 0:
            return 0.0
        
        # Use binary relevance if scores not provided
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}
        
        def dcg_at_k(items: List[str], scores: Dict[str, float], k: int) -> float:
            """Calculate DCG@K"""
            dcg = 0.0
            for i, item in enumerate(items[:k]):
                if item in scores:
                    rel = scores[item]
                    dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0
            return dcg
        
        # Calculate DCG for recommendations
        dcg = dcg_at_k(recommended, relevance_scores, k)
        
        # Calculate ideal DCG (IDCG)
        ideal_order = sorted(relevant, key=lambda x: relevance_scores.get(x, 0), reverse=True)
        idcg = dcg_at_k(ideal_order, relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mean_reciprocal_rank(self, recommended: List[str], relevant: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            
        Returns:
            MRR score
        """
        if not recommended or not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        
        for i, item in enumerate(recommended):
            if item in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def hit_rate_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Hit Rate@K (whether at least one relevant item is in top-k)
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K (0 or 1)
        """
        if not recommended or not relevant or k <= 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        return 1.0 if recommended_k.intersection(relevant_set) else 0.0
    
    def coverage(self, all_recommendations: List[List[str]], all_items: List[str]) -> float:
        """
        Calculate catalog coverage
        
        Args:
            all_recommendations: List of recommendation lists for all users
            all_items: List of all available items
            
        Returns:
            Coverage score (percentage of catalog recommended)
        """
        if not all_items:
            return 0.0
        
        recommended_items = set()
        for user_recs in all_recommendations:
            recommended_items.update(user_recs)
        
        return len(recommended_items) / len(all_items)
    
    def diversity(
        self,
        recommendations: List[str],
        similarity_matrix: Optional[np.ndarray] = None,
        item_features: Optional[Dict[str, List]] = None
    ) -> float:
        """
        Calculate diversity of recommendations
        
        Args:
            recommendations: List of recommended items
            similarity_matrix: Precomputed similarity matrix between items
            item_features: Item features for computing similarity
            
        Returns:
            Diversity score (higher is more diverse)
        """
        if len(recommendations) < 2:
            return 1.0
        
        if similarity_matrix is not None:
            # Use precomputed similarity matrix
            total_similarity = 0.0
            pairs = 0
            
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    # This would require mapping item IDs to matrix indices
                    # Simplified version
                    total_similarity += 0.5  # Placeholder
                    pairs += 1
            
            if pairs == 0:
                return 1.0
            
            avg_similarity = total_similarity / pairs
            return 1.0 - avg_similarity
        
        elif item_features is not None:
            # Calculate diversity based on features
            total_similarity = 0.0
            pairs = 0
            
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    item_i = recommendations[i]
                    item_j = recommendations[j]
                    
                    if item_i in item_features and item_j in item_features:
                        features_i = set(item_features[item_i])
                        features_j = set(item_features[item_j])
                        
                        if features_i or features_j:
                            jaccard_sim = len(features_i.intersection(features_j)) / len(features_i.union(features_j))
                            total_similarity += jaccard_sim
                            pairs += 1
            
            if pairs == 0:
                return 1.0
            
            avg_similarity = total_similarity / pairs
            return 1.0 - avg_similarity
        
        else:
            # Without similarity information, assume maximum diversity
            return 1.0
    
    def novelty(
        self,
        recommendations: List[str],
        item_popularity: Dict[str, float]
    ) -> float:
        """
        Calculate novelty of recommendations
        
        Args:
            recommendations: List of recommended items
            item_popularity: Dictionary mapping items to popularity scores
            
        Returns:
            Novelty score (higher is more novel)
        """
        if not recommendations:
            return 0.0
        
        total_novelty = 0.0
        valid_items = 0
        
        for item in recommendations:
            if item in item_popularity:
                popularity = item_popularity[item]
                # Novelty is inverse of popularity
                novelty = 1.0 - popularity if popularity <= 1.0 else 1.0 / (popularity + 1)
                total_novelty += novelty
                valid_items += 1
        
        return total_novelty / max(valid_items, 1)
    
    def serendipity(
        self,
        recommendations: List[str],
        user_profile: List[str],
        similarity_threshold: float = 0.7
    ) -> float:
        """
        Calculate serendipity of recommendations
        
        Args:
            recommendations: List of recommended items
            user_profile: List of items in user's profile
            similarity_threshold: Threshold for considering items as similar
            
        Returns:
            Serendipity score
        """
        if not recommendations or not user_profile:
            return 0.0
        
        serendipitous_items = 0
        
        for rec_item in recommendations:
            is_serendipitous = True
            
            for profile_item in user_profile:
                # This would require a similarity computation
                # For now, simple check if item is not in user profile
                if rec_item == profile_item:
                    is_serendipitous = False
                    break
            
            if is_serendipitous:
                serendipitous_items += 1
        
        return serendipitous_items / len(recommendations)
    
    def calculate_all_metrics(
        self,
        recommended: List[str],
        relevant: List[str],
        k_values: List[int] = [5, 10, 20],
        item_features: Dict[str, List] = None,
        item_popularity: Dict[str, float] = None,
        user_profile: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for a single user's recommendations
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k_values: Values of K for evaluation
            item_features: Item features for diversity calculation
            item_popularity: Item popularity for novelty calculation
            user_profile: User profile for serendipity calculation
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Calculate metrics for each k value
        for k in k_values:
            metrics[f'precision_at_{k}'] = self.precision_at_k(recommended, relevant, k)
            metrics[f'recall_at_{k}'] = self.recall_at_k(recommended, relevant, k)
            metrics[f'f1_at_{k}'] = self.f1_at_k(recommended, relevant, k)
            metrics[f'ndcg_at_{k}'] = self.ndcg_at_k(recommended, relevant, k)
            metrics[f'hit_rate_at_{k}'] = self.hit_rate_at_k(recommended, relevant, k)
        
        # Calculate ranking metrics
        metrics['mrr'] = self.mean_reciprocal_rank(recommended, relevant)
        
        # Calculate beyond-accuracy metrics
        if item_features:
            metrics['diversity'] = self.diversity(recommended, item_features=item_features)
        
        if item_popularity:
            metrics['novelty'] = self.novelty(recommended, item_popularity)
        
        if user_profile:
            metrics['serendipity'] = self.serendipity(recommended, user_profile)
        
        return metrics
