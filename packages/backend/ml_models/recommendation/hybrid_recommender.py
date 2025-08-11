import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import asyncio
import logging
from datetime import datetime
import json

from .collaborative_filtering import CollaborativeFilteringEngine
from .content_based_filtering import ContentBasedRecommendationEngine
from ..evaluation.metrics import RecommendationMetrics
from ...core.privacy.differential_privacy import DifferentialPrivacy

logger = logging.getLogger(__name__)

class HybridRecommendationEngine:
    """
    Hybrid recommendation engine combining collaborative and content-based filtering
    with explainability and privacy preservation
    """
    
    def __init__(
        self,
        cf_config: Dict[str, Any] = None,
        cb_config: Dict[str, Any] = None,
        hybrid_config: Dict[str, Any] = None,
        privacy_config: Dict[str, Any] = None
    ):
        # Initialize component engines
        self.cf_engine = CollaborativeFilteringEngine(cf_config, privacy_config)
        self.cb_engine = ContentBasedRecommendationEngine(cb_config)
        
        # Hybrid configuration
        self.hybrid_config = hybrid_config or self._default_hybrid_config()
        
        # Privacy preservation
        self.privacy_config = privacy_config or {"epsilon": 0.5, "delta": 1e-5}
        self.dp_engine = DifferentialPrivacy(
            epsilon=self.privacy_config["epsilon"],
            delta=self.privacy_config["delta"]
        )
        
        # Evaluation metrics
        self.metrics = RecommendationMetrics()
        
        # Model states
        self.cf_trained = False
        self.cb_fitted = False
        self.is_ready = False
        
        # Explanation components
        self.explanation_templates = self._init_explanation_templates()
        
        # A/B testing support
        self.ab_test_groups = ['control', 'cf_heavy', 'cb_heavy', 'balanced']
        
    def _default_hybrid_config(self) -> Dict[str, Any]:
        """Default hybrid configuration"""
        return {
            'combination_method': 'weighted_average',  # 'weighted_average', 'rank_fusion', 'stacking'
            'cf_weight': 0.6,
            'cb_weight': 0.4,
            'min_cf_confidence': 0.3,
            'min_cb_confidence': 0.2,
            'diversity_penalty': 0.1,
            'novelty_boost': 0.15,
            'explanation_depth': 'detailed',  # 'simple', 'detailed', 'technical'
            'enable_cold_start_handling': True,
            'cold_start_cb_weight': 0.8,
            'popularity_fallback_threshold': 0.1
        }
    
    def _init_explanation_templates(self) -> Dict[str, str]:
        """Initialize explanation templates for different recommendation scenarios"""
        return {
            'collaborative': "Users who liked similar apps to you also enjoyed {app_name}",
            'content_based': "{app_name} is recommended because it's similar to apps you've used",
            'hybrid': "{app_name} is recommended based on both user preferences and content similarity",
            'cold_start': "{app_name} is popular among users with similar interests",
            'diverse': "{app_name} offers something different from your usual preferences",
            'novel': "{app_name} is a new app that matches your interests"
        }
    
    async def train(
        self,
        interactions: List[Dict[str, Any]],
        apps_data: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train both collaborative and content-based components
        
        Args:
            interactions: User-item interaction data
            apps_data: App content and feature data
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        logger.info("Training hybrid recommendation system")
        
        training_results = {}
        
        # Train collaborative filtering
        try:
            logger.info("Training collaborative filtering component")
            cf_results = await self.cf_engine.train_model(interactions, validation_split)
            training_results['collaborative_filtering'] = cf_results
            self.cf_trained = True
            logger.info("Collaborative filtering training completed")
        except Exception as e:
            logger.error(f"Collaborative filtering training failed: {e}")
            training_results['collaborative_filtering'] = {'error': str(e)}
        
        # Train content-based filtering
        try:
            logger.info("Training content-based filtering component")
            cb_results = await self.cb_engine.fit(apps_data)
            training_results['content_based_filtering'] = cb_results
            self.cb_fitted = True
            logger.info("Content-based filtering training completed")
        except Exception as e:
            logger.error(f"Content-based filtering training failed: {e}")
            training_results['content_based_filtering'] = {'error': str(e)}
        
        # System is ready if at least one component is trained
        self.is_ready = self.cf_trained or self.cb_fitted
        
        if self.is_ready:
            # Validate hybrid system
            validation_results = await self._validate_hybrid_system(interactions, apps_data)
            training_results['hybrid_validation'] = validation_results
        
        training_results['system_status'] = {
            'cf_trained': self.cf_trained,
            'cb_fitted': self.cb_fitted,
            'is_ready': self.is_ready,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Hybrid system training completed. Ready: {self.is_ready}")
        return training_results
    
    async def _validate_hybrid_system(
        self,
        interactions: List[Dict[str, Any]],
        apps_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the hybrid system performance"""
        validation_results = {}
        
        # Sample validation - generate recommendations for a few users
        sample_users = list(set([interaction['user_id'] for interaction in interactions[:100]]))[:5]
        
        for user_id in sample_users:
            try:
                recommendations = await self.recommend(
                    user_id=user_id,
                    num_recommendations=10,
                    return_explanations=True
                )
                validation_results[user_id] = {
                    'num_recommendations': len(recommendations),
                    'has_explanations': any('explanation' in rec for rec in recommendations)
                }
            except Exception as e:
                validation_results[user_id] = {'error': str(e)}
        
        return validation_results
    
    async def recommend(
        self,
        user_id: str,
        num_recommendations: int = 10,
        return_explanations: bool = True,
        diversity_threshold: float = 0.7,
        ab_test_group: str = 'balanced',
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations with explanations
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to return
            return_explanations: Whether to include explanations
            diversity_threshold: Minimum diversity threshold
            ab_test_group: A/B test group configuration
            context: Additional context information
            
        Returns:
            List of recommendations with scores and explanations
        """
        if not self.is_ready:
            raise ValueError("Hybrid system must be trained before generating recommendations")
        
        context = context or {}
        
        # Determine weights based on A/B test group and user characteristics
        cf_weight, cb_weight = self._get_dynamic_weights(user_id, ab_test_group, context)
        
        # Get recommendations from both components
        cf_recommendations = []
        cb_recommendations = []
        
        # Collaborative filtering recommendations
        if self.cf_trained:
            try:
                cf_recommendations = await self.cf_engine.recommend(
                    user_id=user_id,
                    num_recommendations=num_recommendations * 2,  # Get more for fusion
                    diversity_threshold=diversity_threshold
                )
                # Add source information
                for rec in cf_recommendations:
                    rec['source'] = 'collaborative'
                    rec['source_confidence'] = rec.get('confidence', 0.5)
            except Exception as e:
                logger.warning(f"Collaborative filtering failed for user {user_id}: {e}")
        
        # Content-based recommendations
        if self.cb_fitted:
            try:
                # Get user preferences for content-based filtering
                user_preferences = await self._get_user_preferences(user_id, context)
                cb_recommendations = await self.cb_engine.recommend_for_user_preferences(
                    user_preferences=user_preferences,
                    num_recommendations=num_recommendations * 2
                )
                # Add source information
                for rec in cb_recommendations:
                    rec['source'] = 'content_based'
                    rec['source_confidence'] = rec.get('similarity_score', 0.5)
            except Exception as e:
                logger.warning(f"Content-based filtering failed for user {user_id}: {e}")
        
        # Combine recommendations
        hybrid_recommendations = await self._combine_recommendations(
            cf_recommendations,
            cb_recommendations,
            cf_weight,
            cb_weight,
            num_recommendations
        )
        
        # Apply diversity filtering
        diverse_recommendations = self._apply_diversity_filter(
            hybrid_recommendations,
            diversity_threshold,
            num_recommendations
        )
        
        # Add explanations if requested
        if return_explanations:
            diverse_recommendations = await self._add_explanations(
                diverse_recommendations,
                user_id,
                context
            )
        
        # Apply privacy preservation
        diverse_recommendations = self._apply_privacy_preservation(diverse_recommendations)
        
        # Log recommendation for A/B testing
        await self._log_recommendation(
            user_id,
            diverse_recommendations,
            ab_test_group,
            context
        )
        
        return diverse_recommendations[:num_recommendations]
    
    def _get_dynamic_weights(
        self,
        user_id: str,
        ab_test_group: str,
        context: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Determine dynamic weights for CF and CB based on user characteristics and A/B testing
        
        Args:
            user_id: User identifier
            ab_test_group: A/B test group
            context: Additional context
            
        Returns:
            Tuple of (cf_weight, cb_weight)
        """
        # Base weights from configuration
        base_cf_weight = self.hybrid_config['cf_weight']
        base_cb_weight = self.hybrid_config['cb_weight']
        
        # Adjust weights based on A/B test group
        if ab_test_group == 'cf_heavy':
            cf_weight, cb_weight = 0.8, 0.2
        elif ab_test_group == 'cb_heavy':
            cf_weight, cb_weight = 0.2, 0.8
        elif ab_test_group == 'control':
            cf_weight, cb_weight = 0.5, 0.5
        else:  # balanced
            cf_weight, cb_weight = base_cf_weight, base_cb_weight
        
        # Check if this is a cold start user
        is_cold_start = self._is_cold_start_user(user_id)
        if is_cold_start and self.hybrid_config['enable_cold_start_handling']:
            # Favor content-based for cold start users
            cf_weight *= (1 - self.hybrid_config['cold_start_cb_weight'])
            cb_weight = self.hybrid_config['cold_start_cb_weight']
        
        # Normalize weights
        total_weight = cf_weight + cb_weight
        if total_weight > 0:
            cf_weight /= total_weight
            cb_weight /= total_weight
        
        return cf_weight, cb_weight
    
    def _is_cold_start_user(self, user_id: str) -> bool:
        """Check if user is a cold start case"""
        # This would check user interaction history in the database
        # For now, simple check based on user encoder
        if hasattr(self.cf_engine, 'user_encoder'):
            return user_id not in self.cf_engine.user_encoder
        return True
    
    async def _get_user_preferences(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract user preferences from interaction history and context
        
        Args:
            user_id: User identifier
            context: Additional context
            
        Returns:
            User preferences dictionary
        """
        # This would query the database for user preferences
        # For now, return default preferences
        default_preferences = {
            'preferred_categories': context.get('categories', []),
            'rating_threshold': 4.0,
            'novelty_preference': 0.5,
            'diversity_preference': 0.7
        }
        
        return default_preferences
    
    async def _combine_recommendations(
        self,
        cf_recs: List[Dict[str, Any]],
        cb_recs: List[Dict[str, Any]],
        cf_weight: float,
        cb_weight: float,
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """
        Combine recommendations from CF and CB components
        
        Args:
            cf_recs: Collaborative filtering recommendations
            cb_recs: Content-based recommendations
            cf_weight: Weight for CF recommendations
            cb_weight: Weight for CB recommendations
            num_recommendations: Target number of recommendations
            
        Returns:
            Combined recommendations
        """
        # Create a map of all unique items
        all_items = {}
        
        # Process CF recommendations
        for rec in cf_recs:
            item_id = rec.get('app_id') or rec.get('item_id')
            if item_id:
                all_items[item_id] = {
                    'app_id': item_id,
                    'cf_score': rec.get('predicted_rating', rec.get('confidence', 0)),
                    'cf_rank': rec.get('rank', 999),
                    'cf_present': True,
                    'cb_score': 0,
                    'cb_rank': 999,
                    'cb_present': False,
                    'sources': ['collaborative']
                }
        
        # Process CB recommendations
        for rec in cb_recs:
            item_id = rec.get('app_id') or rec.get('item_id')
            if item_id:
                if item_id in all_items:
                    all_items[item_id]['cb_score'] = rec.get('similarity_score', rec.get('confidence', 0))
                    all_items[item_id]['cb_rank'] = rec.get('rank', 999)
                    all_items[item_id]['cb_present'] = True
                    all_items[item_id]['sources'].append('content_based')
                else:
                    all_items[item_id] = {
                        'app_id': item_id,
                        'cf_score': 0,
                        'cf_rank': 999,
                        'cf_present': False,
                        'cb_score': rec.get('similarity_score', rec.get('confidence', 0)),
                        'cb_rank': rec.get('rank', 999),
                        'cb_present': True,
                        'sources': ['content_based']
                    }
        
        # Combine scores based on method
        combined_recs = []
        for item_id, item_data in all_items.items():
            if self.hybrid_config['combination_method'] == 'weighted_average':
                # Normalize scores to [0, 1] range
                cf_score_norm = min(1.0, item_data['cf_score'] / 5.0) if item_data['cf_present'] else 0
                cb_score_norm = item_data['cb_score'] if item_data['cb_present'] else 0
                
                combined_score = (cf_weight * cf_score_norm) + (cb_weight * cb_score_norm)
                
            elif self.hybrid_config['combination_method'] == 'rank_fusion':
                # Reciprocal rank fusion
                cf_rank_score = 1.0 / (item_data['cf_rank'] + 1) if item_data['cf_present'] else 0
                cb_rank_score = 1.0 / (item_data['cb_rank'] + 1) if item_data['cb_present'] else 0
                
                combined_score = (cf_weight * cf_rank_score) + (cb_weight * cb_rank_score)
            
            else:  # Default to weighted average
                cf_score_norm = min(1.0, item_data['cf_score'] / 5.0) if item_data['cf_present'] else 0
                cb_score_norm = item_data['cb_score'] if item_data['cb_present'] else 0
                combined_score = (cf_weight * cf_score_norm) + (cb_weight * cb_score_norm)
            
            combined_recs.append({
                'app_id': item_id,
                'hybrid_score': combined_score,
                'cf_score': item_data['cf_score'],
                'cb_score': item_data['cb_score'],
                'sources': item_data['sources'],
                'confidence': combined_score,
                'recommendation_type': 'hybrid'
            })
        
        # Sort by combined score
        combined_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return combined_recs
    
    def _apply_diversity_filter(
        self,
        recommendations: List[Dict[str, Any]],
        diversity_threshold: float,
        max_items: int
    ) -> List[Dict[str, Any]]:
        """Apply diversity filtering to recommendations"""
        if not recommendations:
            return []
        
        diverse_recs = [recommendations[0]]  # Start with highest scored item
        
        for candidate in recommendations[1:]:
            if len(diverse_recs) >= max_items:
                break
            
            # Check diversity against selected recommendations
            is_diverse = True
            for selected in diverse_recs:
                # Simple diversity check - in practice, would use semantic similarity
                if candidate['app_id'] == selected['app_id']:
                    is_diverse = False
                    break
                
                # Could add category-based diversity checks here
                # if same_category(candidate, selected) and diversity_score < threshold:
                #     is_diverse = False
                #     break
            
            if is_diverse:
                diverse_recs.append(candidate)
        
        return diverse_recs
    
    async def _add_explanations(
        self,
        recommendations: List[Dict[str, Any]],
        user_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Add explanations to recommendations
        
        Args:
            recommendations: List of recommendations
            user_id: User identifier
            context: Additional context
            
        Returns:
            Recommendations with explanations
        """
        for rec in recommendations:
            explanation = await self._generate_explanation(rec, user_id, context)
            rec['explanation'] = explanation
        
        return recommendations
    
    async def _generate_explanation(
        self,
        recommendation: Dict[str, Any],
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single recommendation
        
        Args:
            recommendation: Recommendation dictionary
            user_id: User identifier  
            context: Additional context
            
        Returns:
            Explanation dictionary
        """
        sources = recommendation.get('sources', [])
        app_id = recommendation['app_id']
        
        # Determine primary explanation type
        if 'collaborative' in sources and 'content_based' in sources:
            explanation_type = 'hybrid'
        elif 'collaborative' in sources:
            explanation_type = 'collaborative'
        elif 'content_based' in sources:
            explanation_type = 'content_based'
        else:
            explanation_type = 'hybrid'
        
        # Generate explanation components
        explanation = {
            'type': explanation_type,
            'confidence': recommendation.get('confidence', 0.5),
            'primary_reason': self.explanation_templates.get(explanation_type, 'Recommended for you'),
            'supporting_reasons': [],
            'technical_details': {}
        }
        
        # Add detailed explanations based on depth setting
        if self.hybrid_config['explanation_depth'] in ['detailed', 'technical']:
            if 'collaborative' in sources:
                explanation['supporting_reasons'].append(
                    f"Similar users liked this app (CF score: {recommendation.get('cf_score', 0):.2f})"
                )
            
            if 'content_based' in sources:
                explanation['supporting_reasons'].append(
                    f"Matches your app preferences (CB score: {recommendation.get('cb_score', 0):.2f})"
                )
        
        if self.hybrid_config['explanation_depth'] == 'technical':
            explanation['technical_details'] = {
                'hybrid_score': recommendation.get('hybrid_score', 0),
                'cf_score': recommendation.get('cf_score', 0),
                'cb_score': recommendation.get('cb_score', 0),
                'sources': sources,
                'algorithm_version': '1.0'
            }
        
        return explanation
    
    def _apply_privacy_preservation(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply privacy preservation to recommendations"""
        # Add minimal noise to scores to protect privacy
        for rec in recommendations:
            if 'hybrid_score' in rec:
                noise = np.random.laplace(0, 0.01)  # Small noise
                rec['hybrid_score'] = max(0, rec['hybrid_score'] + noise)
        
        return recommendations
    
    async def _log_recommendation(
        self,
        user_id: str,
        recommendations: List[Dict[str, Any]],
        ab_test_group: str,
        context: Dict[str, Any]
    ):
        """Log recommendation for A/B testing and monitoring"""
        log_entry = {
            'user_id': user_id,
            'recommendations': [
                {
                    'app_id': rec['app_id'],
                    'hybrid_score': rec.get('hybrid_score', 0),
                    'sources': rec.get('sources', [])
                }
                for rec in recommendations
            ],
            'ab_test_group': ab_test_group,
            'context': context,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': '1.0'
        }
        
        # In practice, this would be stored in the database
        logger.info(f"Recommendation logged for user {user_id}, group {ab_test_group}")
    
    async def evaluate_recommendations(
        self,
        test_interactions: List[Dict[str, Any]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Evaluate the hybrid recommendation system
        
        Args:
            test_interactions: Test interaction data
            k_values: Values of K for evaluation metrics
            
        Returns:
            Evaluation results
        """
        if not self.is_ready:
            raise ValueError("System must be trained before evaluation")
        
        logger.info("Starting hybrid system evaluation")
        
        evaluation_results = {}
        
        # Group interactions by user
        user_interactions = {}
        for interaction in test_interactions:
            user_id = interaction['user_id']
            if user_id not in user_interactions:
                user_interactions[user_id] = []
            user_interactions[user_id].append(interaction)
        
        # Sample users for evaluation
        sample_users = list(user_interactions.keys())[:100]  # Limit for performance
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_id in sample_users:
                try:
                    # Get ground truth items
                    true_items = [
                        interaction['app_id'] 
                        for interaction in user_interactions[user_id]
                        if interaction.get('rating', 1) >= 4  # Relevant threshold
                    ]
                    
                    if not true_items:
                        continue
                    
                    # Get recommendations
                    recommendations = await self.recommend(
                        user_id=user_id,
                        num_recommendations=k,
                        return_explanations=False
                    )
                    
                    recommended_items = [rec['app_id'] for rec in recommendations]
                    
                    # Calculate metrics
                    precision = self.metrics.precision_at_k(recommended_items, true_items, k)
                    recall = self.metrics.recall_at_k(recommended_items, true_items, k)
                    ndcg = self.metrics.ndcg_at_k(recommended_items, true_items, k)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    ndcg_scores.append(ndcg)
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed for user {user_id}: {e}")
                    continue
            
            # Aggregate results
            evaluation_results[f'precision_at_{k}'] = np.mean(precision_scores) if precision_scores else 0
            evaluation_results[f'recall_at_{k}'] = np.mean(recall_scores) if recall_scores else 0
            evaluation_results[f'ndcg_at_{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0
        
        evaluation_results['num_users_evaluated'] = len(sample_users)
        evaluation_results['evaluation_timestamp'] = datetime.utcnow().isoformat()
        
        logger.info("Hybrid system evaluation completed")
        return evaluation_results
    
    async def save_model(self, save_dir: str):
        """Save the hybrid model components"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save CF component
        if self.cf_trained:
            cf_path = os.path.join(save_dir, 'collaborative_filtering.pt')
            await self.cf_engine._save_model(cf_path)
        
        # Save CB component
        if self.cb_fitted:
            cb_path = os.path.join(save_dir, 'content_based_filtering.pkl')
            await self.cb_engine.save_model(cb_path)
        
        # Save hybrid configuration
        config_path = os.path.join(save_dir, 'hybrid_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'hybrid_config': self.hybrid_config,
                'privacy_config': self.privacy_config,
                'cf_trained': self.cf_trained,
                'cb_fitted': self.cb_fitted,
                'is_ready': self.is_ready
            }, f, indent=2)
        
        logger.info(f"Hybrid model saved to {save_dir}")
    
    async def load_model(self, load_dir: str):
        """Load the hybrid model components"""
        import os
        
        # Load configuration
        config_path = os.path.join(load_dir, 'hybrid_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.hybrid_config = config_data['hybrid_config']
                self.privacy_config = config_data['privacy_config']
                self.cf_trained = config_data['cf_trained']
                self.cb_fitted = config_data['cb_fitted']
                self.is_ready = config_data['is_ready']
        
        # Load CF component
        cf_path = os.path.join(load_dir, 'collaborative_filtering.pt')
        if os.path.exists(cf_path) and self.cf_trained:
            await self.cf_engine.load_model(cf_path)
        
        # Load CB component
        cb_path = os.path.join(load_dir, 'content_based_filtering.pkl')
        if os.path.exists(cb_path) and self.cb_fitted:
            await self.cb_engine.load_model(cb_path)
        
        logger.info(f"Hybrid model loaded from {load_dir}")

# Example usage and testing
if __name__ == "__main__":
    async def test_hybrid_system():
        # Sample data
        interactions = [
            {"user_id": "user1", "app_id": "app1", "rating": 5, "interaction_type": "rate"},
            {"user_id": "user1", "app_id": "app2", "rating": 3, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app1", "rating": 4, "interaction_type": "rate"},
        ]
        
        apps_data = [
            {
                'id': 'app1',
                'name': 'Photo Editor',
                'description': 'Edit photos with filters',
                'category': 'Photography',
                'tags': ['photo', 'editor'],
                'security_score': 85,
                'rating_average': 4.5,
                'downloads_count': 100000
            }
        ]
        
        # Initialize and train
        hybrid_engine = HybridRecommendationEngine()
        training_results = await hybrid_engine.train(interactions, apps_data)
        print("Training results:", training_results)
        
        # Generate recommendations
        recommendations = await hybrid_engine.recommend(
            user_id="user1",
            num_recommendations=5,
            return_explanations=True
        )
        print("Recommendations:", recommendations)
    
    asyncio.run(test_hybrid_system())
