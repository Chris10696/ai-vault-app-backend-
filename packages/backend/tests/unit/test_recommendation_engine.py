import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock
import tempfile
import os

from ...ml_models.recommendation.hybrid_recommender import HybridRecommendationEngine
from ...ml_models.recommendation.collaborative_filtering import CollaborativeFilteringEngine
from ...ml_models.recommendation.content_based_filtering import ContentBasedRecommendationEngine


class TestCollaborativeFiltering:
    """Test cases for collaborative filtering engine"""

    @pytest.fixture
    def sample_interactions(self):
        return [
            {"user_id": "user1", "app_id": "app1", "rating": 5, "interaction_type": "rate"},
            {"user_id": "user1", "app_id": "app2", "rating": 3, "interaction_type": "rate"},
            {"user_id": "user1", "app_id": "app3", "rating": 4, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app1", "rating": 4, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app2", "rating": 2, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app4", "rating": 5, "interaction_type": "rate"},
            {"user_id": "user3", "app_id": "app1", "rating": 3, "interaction_type": "rate"},
            {"user_id": "user3", "app_id": "app3", "rating": 5, "interaction_type": "rate"},
            {"user_id": "user3", "app_id": "app4", "rating": 4, "interaction_type": "rate"},
        ]

    @pytest.fixture
    def cf_engine(self):
        return CollaborativeFilteringEngine(device="cpu")

    @pytest.mark.asyncio
    async def test_data_preparation(self, cf_engine, sample_interactions):
        """Test data preparation process"""
        user_tensor, item_tensor, rating_tensor = await cf_engine.prepare_data(sample_interactions)

        assert user_tensor is not None
        assert item_tensor is not None
        assert rating_tensor is not None
        assert len(user_tensor) == len(sample_interactions)
        assert len(cf_engine.user_encoder) == 3  # 3 unique users
        assert len(cf_engine.item_encoder) == 4  # 4 unique items

    @pytest.mark.asyncio
    async def test_model_training(self, cf_engine, sample_interactions):
        """Test model training process"""
        cf_engine.model_config['epochs'] = 5
        cf_engine.model_config['batch_size'] = 32

        training_results = await cf_engine.train_model(sample_interactions, validation_split=0.2)

        assert training_results is not None
        assert 'training_history' in training_results
        assert 'final_metrics' in training_results
        assert cf_engine.is_trained
        assert cf_engine.model is not None

    @pytest.mark.asyncio
    async def test_predictions(self, cf_engine, sample_interactions):
        """Test prediction generation"""
        cf_engine.model_config['epochs'] = 3
        await cf_engine.train_model(sample_interactions, validation_split=0.2)

        predictions = await cf_engine.predict("user1", ["app1", "app4"])

        assert isinstance(predictions, list)
        assert len(predictions) <= 2
        if predictions:
            assert 'item_id' in predictions[0]
            assert 'predicted_rating' in predictions[0]
            assert 'confidence' in predictions[0]

    @pytest.mark.asyncio
    async def test_recommendations(self, cf_engine, sample_interactions):
        """Test recommendation generation"""
        cf_engine.model_config['epochs'] = 3
        await cf_engine.train_model(sample_interactions, validation_split=0.2)

        recommendations = await cf_engine.recommend("user1", num_recommendations=3)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        if recommendations:
            assert 'item_id' in recommendations[0]
            assert 'predicted_rating' in recommendations[0]

    @pytest.mark.asyncio
    async def test_cold_start_handling(self, cf_engine, sample_interactions):
        """Test cold start user handling"""
        cf_engine.model_config['epochs'] = 3
        await cf_engine.train_model(sample_interactions, validation_split=0.2)

        recommendations = await cf_engine.recommend("new_user", num_recommendations=3)

        assert isinstance(recommendations, list)


class TestContentBasedFiltering:
    """Test cases for content-based filtering engine"""

    @pytest.fixture
    def sample_apps(self):
        return [
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
            },
            {
                'id': 'app3',
                'name': 'Task Manager',
                'description': 'Organize your tasks and boost productivity',
                'category': 'Productivity',
                'tags': ['tasks', 'productivity', 'organize'],
                'security_score': 95,
                'rating_average': 4.0,
                'downloads_count': 50000
            },
            {
                'id': 'app4',
                'name': 'Video Editor',
                'description': 'Edit videos with professional tools',
                'category': 'Multimedia',
                'tags': ['video', 'editor', 'multimedia'],
                'security_score': 80,
                'rating_average': 4.3,
                'downloads_count': 120000
            }
        ]

    @pytest.fixture
    def cb_engine(self):
        return ContentBasedRecommendationEngine()

    @pytest.mark.asyncio
    async def test_model_fitting(self, cb_engine, sample_apps):
        """Test content-based model fitting"""
        fitting_stats = await cb_engine.fit(sample_apps)

        assert fitting_stats is not None
        assert 'num_apps' in fitting_stats
        assert 'feature_dimensions' in fitting_stats
        assert fitting_stats['num_apps'] == 4
        assert cb_engine.is_fitted
        assert cb_engine.content_features is not None
        assert cb_engine.similarity_matrix is not None

    @pytest.mark.asyncio
    async def test_similar_apps(self, cb_engine, sample_apps):
        """Test finding similar apps"""
        await cb_engine.fit(sample_apps)

        similar_apps = await cb_engine.get_similar_apps("app1", num_recommendations=2)

        assert isinstance(similar_apps, list)
        assert len(similar_apps) <= 2
        if similar_apps:
            assert 'app_id' in similar_apps[0]
            assert 'similarity_score' in similar_apps[0]
            assert 'explanation' in similar_apps[0]

    @pytest.mark.asyncio
    async def test_user_preference_recommendations(self, cb_engine, sample_apps):
        """Test recommendations based on user preferences"""
        await cb_engine.fit(sample_apps)

        user_preferences = {
            'preferred_categories': ['Photography'],
            'min_rating': 4.0
        }

        recommendations = await cb_engine.recommend_for_user_preferences(
            user_preferences, num_recommendations=3
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3


class TestHybridRecommender:
    """Test cases for hybrid recommendation engine"""

    @pytest.fixture
    def sample_interactions(self):
        return [
            {"user_id": "user1", "app_id": "app1", "rating": 5, "interaction_type": "rate"},
            {"user_id": "user1", "app_id": "app2", "rating": 3, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app1", "rating": 4, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app3", "rating": 5, "interaction_type": "rate"},
        ]

    @pytest.fixture
    def sample_apps(self):
        return [
            {
                'id': 'app1',
                'name': 'Test App 1',
                'description': 'A test application',
                'category': 'Test',
                'tags': ['test'],
                'security_score': 85,
                'rating_average': 4.5,
                'downloads_count': 100000
            },
            {
                'id': 'app2',
                'name': 'Test App 2',
                'description': 'Another test application',
                'category': 'Test',
                'tags': ['test'],
                'security_score': 80,
                'rating_average': 4.0,
                'downloads_count': 50000
            }
        ]

    @pytest.fixture
    def hybrid_engine(self):
        cf_config = {"epochs": 3, "batch_size": 32}
        return HybridRecommendationEngine(cf_config=cf_config)

    @pytest.mark.asyncio
    async def test_hybrid_training(self, hybrid_engine, sample_interactions, sample_apps):
        """Test hybrid system training"""
        training_results = await hybrid_engine.train(
            sample_interactions,
            sample_apps,
            validation_split=0.2
        )

        assert training_results is not None
        assert 'system_status' in training_results
        status = training_results['system_status']
        assert status['cf_trained'] or status['cb_fitted']
        assert status['is_ready']

    @pytest.mark.asyncio
    async def test_hybrid_recommendations(self, hybrid_engine, sample_interactions, sample_apps):
        """Test hybrid recommendation generation"""
        await hybrid_engine.train(sample_interactions, sample_apps, validation_split=0.2)

        recommendations = await hybrid_engine.recommend(
            user_id="user1",
            num_recommendations=3,
            return_explanations=True,
            ab_test_group="balanced"
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        if recommendations:
            assert 'app_id' in recommendations[0]
            assert 'hybrid_score' in recommendations[0] or 'confidence' in recommendations[0]
            assert 'explanation' in recommendations[0]

    @pytest.mark.asyncio
    async def test_model_persistence(self, hybrid_engine, sample_interactions, sample_apps):
        """Test hybrid model saving and loading"""
        await hybrid_engine.train(sample_interactions, sample_apps, validation_split=0.2)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "hybrid_model")
            await hybrid_engine.save_model(save_path)

            assert os.path.exists(os.path.join(save_path, "hybrid_config.json"))

            new_engine = HybridRecommendationEngine()
            await new_engine.load_model(save_path)

            assert new_engine.is_ready == hybrid_engine.is_ready
            assert new_engine.cf_trained == hybrid_engine.cf_trained
            assert new_engine.cb_fitted == hybrid_engine.cb_fitted


class TestRecommendationAPI:
    """Test cases for recommendation API endpoints"""

    @pytest.fixture
    def client(self):
        from ...services.discovery.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": "Bearer mock_jwt_token"}

    def test_model_status_endpoint(self, client):
        response = client.get("/api/v1/recommendations/status")
        assert response.status_code == 200
        data = response.json()
        assert "cf_trained" in data
        assert "cb_fitted" in data
        assert "hybrid_ready" in data
        assert "model_version" in data

    def test_health_check_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "models" in data

    def test_recommendations_endpoint_unauthorized(self, client):
        response = client.post("/api/v1/recommendations/recommend",
                               json={"user_id": "test_user", "num_recommendations": 5})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_recommendations_endpoint_authorized(self, client, auth_headers):
        pass


class TestPerformance:
    """Performance tests for recommendation system"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_recommendation_latency(self):
        import time

        interactions = []
        for user_id in range(100):
            for app_id in range(50):
                if np.random.random() > 0.8:
                    interactions.append({
                        "user_id": f"user_{user_id}",
                        "app_id": f"app_{app_id}",
                        "rating": np.random.randint(1, 6),
                        "interaction_type": "rate"
                    })

        hybrid_engine = HybridRecommendationEngine()
        await hybrid_engine.train(interactions, [], validation_split=0.2)

        start_time = time.time()
        recommendations = await hybrid_engine.recommend("user_1", num_recommendations=10)
        end_time = time.time()

        latency = end_time - start_time
        assert latency < 1.0
        assert len(recommendations) <= 10

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_recommendations(self):
        hybrid_engine = HybridRecommendationEngine()

        interactions = [
            {"user_id": "user1", "app_id": "app1", "rating": 5, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app1", "rating": 4, "interaction_type": "rate"},
        ]

        await hybrid_engine.train(interactions, [], validation_split=0.2)

        tasks = []
        for i in range(10):
            task = hybrid_engine.recommend(f"user_{i}", num_recommendations=5)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_recommendation_pipeline(self):
        pass

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_integration(self):
        pass


if __name__ == "__main__":
    pytest.main(["-v", __file__])
