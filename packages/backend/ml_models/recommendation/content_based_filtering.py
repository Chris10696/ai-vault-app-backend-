import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import asyncio
import logging
import pickle
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ContentBasedRecommendationEngine:
    """
    Content-based filtering engine for app recommendations
    Uses app features, descriptions, and metadata for similarity computation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Feature extractors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config['tfidf_max_features'],
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.mlb_categories = MultiLabelBinarizer()
        self.mlb_tags = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        
        # Computed features
        self.content_features = None
        self.similarity_matrix = None
        self.app_index_map = {}
        self.index_app_map = {}
        
        self.is_fitted = False
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for content-based filtering"""
        return {
            'tfidf_max_features': 5000,
            'similarity_threshold': 0.1,
            'feature_weights': {
                'description': 0.4,
                'category': 0.2,
                'tags': 0.2,
                'features': 0.1,
                'ratings': 0.1
            },
            'diversity_penalty': 0.1
        }
    
    async def fit(self, apps_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fit the content-based model on app data
        
        Args:
            apps_data: List of app dictionaries with features
            
        Returns:
            Fitting statistics
        """
        logger.info(f"Fitting content-based model on {len(apps_data)} apps")
        
        if not apps_data:
            raise ValueError("No app data provided for fitting")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(apps_data)
        
        # Create app index mappings
        self.app_index_map = {app_id: idx for idx, app_id in enumerate(df['id'])}
        self.index_app_map = {idx: app_id for app_id, idx in self.app_index_map.items()}
        
        # Extract and process features
        feature_components = []
        
        # 1. Text features from descriptions
        descriptions = df['description'].fillna('').astype(str)
        tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions)
        feature_components.append(('tfidf', tfidf_features.toarray()))
        
        # 2. Category features
        categories = df['category'].fillna('').apply(lambda x: [x] if x else [])
        category_features = self.mlb_categories.fit_transform(categories)
        feature_components.append(('categories', category_features))
        
        # 3. Tag features
        tags = df['tags'].fillna([]).apply(lambda x: x if isinstance(x, list) else [])
        tag_features = self.mlb_tags.fit_transform(tags)
        feature_components.append(('tags', tag_features))
        
        # 4. Numerical features
        numerical_cols = ['security_score', 'rating_average', 'downloads_count']
        numerical_features = df[numerical_cols].fillna(0).values
        numerical_features = self.scaler.fit_transform(numerical_features)
        feature_components.append(('numerical', numerical_features))
        
        # 5. App feature vectors (if available)
        if 'features' in df.columns:
            feature_vectors = self._process_feature_dict(df['features'])
            if feature_vectors is not None:
                feature_components.append(('app_features', feature_vectors))
        
        # Combine all features with weights
        self.content_features = self._combine_features(feature_components)
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.content_features)
        
        # Zero out self-similarities and apply threshold
        np.fill_diagonal(self.similarity_matrix, 0)
        self.similarity_matrix[self.similarity_matrix < self.config['similarity_threshold']] = 0
        
        self.is_fitted = True
        
        fitting_stats = {
            'num_apps': len(apps_data),
            'feature_dimensions': {
                name: features.shape[1] for name, features in feature_components
            },
            'total_feature_dim': self.content_features.shape[1],
            'avg_similarity': float(np.mean(self.similarity_matrix[self.similarity_matrix > 0])),
            'sparsity': float(np.sum(self.similarity_matrix == 0) / self.similarity_matrix.size)
        }
        
        logger.info(f"Content-based model fitted successfully: {fitting_stats}")
        return fitting_stats
    
    def _process_feature_dict(self, feature_series: pd.Series) -> Optional[np.ndarray]:
        """Process app feature dictionaries into numerical arrays"""
        try:
            # Extract common features across all apps
            all_features = set()
            for features in feature_series:
                if isinstance(features, dict):
                    all_features.update(features.keys())
            
            if not all_features:
                return None
            
            # Create feature matrix
            feature_matrix = []
            for features in feature_series:
                if isinstance(features, dict):
                    feature_vector = [features.get(feat, 0) for feat in sorted(all_features)]
                else:
                    feature_vector = [0] * len(all_features)
                feature_matrix.append(feature_vector)
            
            return np.array(feature_matrix, dtype=float)
            
        except Exception as e:
            logger.warning(f"Failed to process feature dictionaries: {e}")
            return None
    
    def _combine_features(self, feature_components: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Combine different feature types with weights"""
        weighted_features = []
        
        for name, features in feature_components:
            # Get weight for this feature type
            weight = self.config['feature_weights'].get(name, 0.1)
            
            # Normalize features to unit length for each sample
            feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
            feature_norms[feature_norms == 0] = 1  # Avoid division by zero
            normalized_features = features / feature_norms
            
            # Apply weight
            weighted_features.append(normalized_features * weight)
        
        # Concatenate all weighted features
        combined_features = np.hstack(weighted_features)
        
        return combined_features
    
    async def get_similar_apps(
        self,
        app_id: str,
        num_recommendations: int = 10,
        diversity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find apps similar to a given app
        
        Args:
            app_id: Target app ID
            num_recommendations: Number of similar apps to return
            diversity_threshold: Minimum diversity between recommendations
            
        Returns:
            List of similar apps with similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")
        
        if app_id not in self.app_index_map:
            logger.warning(f"App {app_id} not found in fitted data")
            return []
        
        app_idx = self.app_index_map[app_id]
        
        # Get similarity scores for this app
        similarities = self.similarity_matrix[app_idx]
        
        # Get top similar apps
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if len(recommendations) >= num_recommendations:
                break
            
            if similarities[idx] <= 0:  # No more similar apps
                break
            
            similar_app_id = self.index_app_map[idx]
            
            # Check diversity constraint
            if self._is_diverse_enough(similar_app_id, recommendations, diversity_threshold):
                recommendations.append({
                    'app_id': similar_app_id,
                    'similarity_score': float(similarities[idx]),
                    'rank': len(recommendations) + 1,
                    'explanation': self._generate_explanation(app_id, similar_app_id, similarities[idx])
                })
        
        return recommendations
    
    def _is_diverse_enough(
        self,
        candidate_app: str,
        current_recommendations: List[Dict[str, Any]],
        threshold: float
    ) -> bool:
        """Check if candidate app is diverse enough from current recommendations"""
        if not current_recommendations:
            return True
        
        if candidate_app not in self.app_index_map:
            return False
        
        candidate_idx = self.app_index_map[candidate_app]
        
        for rec in current_recommendations:
            rec_app_id = rec['app_id']
            if rec_app_id not in self.app_index_map:
                continue
            
            rec_idx = self.app_index_map[rec_app_id]
            similarity = self.similarity_matrix[candidate_idx, rec_idx]
            
            if similarity > threshold:
                return False  # Too similar to existing recommendation
        
        return True
    
    def _generate_explanation(self, source_app_id: str, target_app_id: str, similarity_score: float) -> Dict[str, Any]:
        """Generate explanation for why apps are similar"""
        # This is a simplified explanation - could be enhanced with feature attribution
        explanation = {
            'similarity_score': float(similarity_score),
            'confidence': 'high' if similarity_score > 0.7 else 'medium' if similarity_score > 0.4 else 'low',
            'reasoning': 'Based on app description, category, and feature similarity'
        }
        
        return explanation
    
    async def recommend_for_user_preferences(
        self,
        user_preferences: Dict[str, Any],
        num_recommendations: int = 10,
        boost_factor: float = 1.2
    ) -> List[Dict[str, Any]]:
        """
        Recommend apps based on user preferences
        
        Args:
            user_preferences: User preference dictionary
            num_recommendations: Number of recommendations
            boost_factor: Boost factor for preferred categories
            
        Returns:
            List of recommended apps
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")
        
        # Create user preference vector
        user_vector = await self._create_user_preference_vector(user_preferences)
        
        # Compute similarity between user preferences and all apps
        similarities = cosine_similarity([user_vector], self.content_features)[0]
        
        # Apply category boosts if specified
        if 'preferred_categories' in user_preferences:
            similarities = self._apply_category_boost(similarities, user_preferences['preferred_categories'], boost_factor)
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            app_id = self.index_app_map[idx]
            recommendations.append({
                'app_id': app_id,
                'similarity_score': float(similarities[idx]),
                'rank': i + 1,
                'recommendation_type': 'content_based',
                'explanation': {
                    'match_score': float(similarities[idx]),
                    'reasoning': 'Based on your preferences and app content similarity'
                }
            })
        
        return recommendations
    
    async def _create_user_preference_vector(self, preferences: Dict[str, Any]) -> np.ndarray:
        """Create a feature vector from user preferences"""
        # Initialize with zero vector
        user_vector = np.zeros(self.content_features.shape[1])
        
        # This is a simplified implementation
        # In practice, you'd map user preferences to the same feature space
        # For now, we'll create a basic vector
        
        return user_vector
    
    def _apply_category_boost(
        self,
        similarities: np.ndarray,
        preferred_categories: List[str],
        boost_factor: float
    ) -> np.ndarray:
        """Apply boost to apps in preferred categories"""
        boosted_similarities = similarities.copy()
        
        # This would need access to app category information
        # Placeholder implementation
        
        return boosted_similarities
    
    async def save_model(self, save_path: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'config': self.config,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'mlb_categories': self.mlb_categories,
            'mlb_tags': self.mlb_tags,
            'scaler': self.scaler,
            'content_features': self.content_features,
            'similarity_matrix': self.similarity_matrix,
            'app_index_map': self.app_index_map,
            'index_app_map': self.index_app_map,
            'is_fitted': self.is_fitted
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Content-based model saved to {save_path}")
    
    async def load_model(self, load_path: str):
        """Load a saved model"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore all attributes
        for key, value in model_data.items():
            setattr(self, key, value)
        
        logger.info(f"Content-based model loaded from {load_path}")

# Example usage
if __name__ == "__main__":
    async def test_content_based():
        # Sample app data
        sample_apps = [
            {
                'id': 'app1',
                'name': 'Photo Editor Pro',
                'description': 'Advanced photo editing with filters and effects',
                'category': 'Photography',
                'tags': ['photo', 'editor', 'filters'],
                'security_score': 85,
                'rating_average': 4.5,
                'downloads_count': 100000
            },
            {
                'id': 'app2',
                'name': 'Camera Plus',
                'description': 'Professional camera app with manual controls',
                'category': 'Photography',
                'tags': ['camera', 'professional', 'manual'],
                'security_score': 90,
                'rating_average': 4.2,
                'downloads_count': 75000
            }
        ]
        
        # Initialize and fit model
        engine = ContentBasedRecommendationEngine()
        fitting_stats = await engine.fit(sample_apps)
        print("Fitting stats:", fitting_stats)
        
        # Get similar apps
        similar_apps = await engine.get_similar_apps('app1', num_recommendations=5)
        print("Similar apps:", similar_apps)
    
    asyncio.run(test_content_based())
