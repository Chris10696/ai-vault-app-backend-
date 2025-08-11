import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import asyncio
import logging
from datetime import datetime
import pickle
import os

from ...core.privacy.differential_privacy import DifferentialPrivacy
from ...core.database.models import UserInteraction, User, App

logger = logging.getLogger(__name__)

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model implementation
    Combines Matrix Factorization with Multi-Layer Perceptron
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_dim: int = 64,
        mlp_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers
        
        # Matrix Factorization Embeddings
        self.user_mf_embedding = nn.Embedding(num_users, mf_dim)
        self.item_mf_embedding = nn.Embedding(num_items, mf_dim)
        
        # MLP Embeddings
        mlp_user_dim = mlp_layers[0] // 2
        mlp_item_dim = mlp_layers // 2
        self.user_mlp_embedding = nn.Embedding(num_users, mlp_user_dim)
        self.item_mlp_embedding = nn.Embedding(num_items, mlp_item_dim)
        
        # MLP Layers
        self.mlp_layers_list = nn.ModuleList()
        input_size = mlp_layers
        
        for layer_size in mlp_layers[1:]:
            self.mlp_layers_list.append(nn.Linear(input_size, layer_size))
            self.mlp_layers_list.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            self.mlp_layers_list.append(nn.Dropout(dropout))
            input_size = layer_size
        
        # Final prediction layer
        self.prediction_layer = nn.Linear(mf_dim + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Matrix Factorization component
        user_mf_vec = self.user_mf_embedding(user_ids)
        item_mf_vec = self.item_mf_embedding(item_ids)
        mf_output = torch.mul(user_mf_vec, item_mf_vec)
        
        # MLP component
        user_mlp_vec = self.user_mlp_embedding(user_ids)
        item_mlp_vec = self.item_mlp_embedding(item_ids)
        mlp_input = torch.cat([user_mlp_vec, item_mlp_vec], dim=-1)
        
        mlp_output = mlp_input
        for layer in self.mlp_layers_list:
            mlp_output = layer(mlp_output)
        
        # Combine MF and MLP
        combined = torch.cat([mf_output, mlp_output], dim=-1)
        prediction = self.prediction_layer(combined)
        
        return torch.sigmoid(prediction) * 5  # Scale to 1-5 rating

class CollaborativeFilteringEngine:
    """
    Main engine for collaborative filtering recommendations with privacy preservation
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any] = None,
        privacy_config: Dict[str, Any] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model_config = model_config or self._default_model_config()
        self.privacy_config = privacy_config or {"epsilon": 0.5, "delta": 1e-5}
        
        self.model = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        self.is_trained = False
        
        # Privacy components
        self.dp_engine = DifferentialPrivacy(
            epsilon=self.privacy_config["epsilon"],
            delta=self.privacy_config["delta"]
        )
        
        # Training history
        self.training_history = {
            "losses": [],
            "val_losses": [],
            "metrics": []
        }
    
    def _default_model_config(self) -> Dict[str, Any]:
        """Default model configuration"""
        return {
            "mf_dim": 64,
            "mlp_layers": [128, 64, 32],
            "dropout": 0.2,
            "activation": "relu",
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 100,
            "weight_decay": 1e-5,
            "early_stopping_patience": 10
        }
    
    async def prepare_data(self, interactions: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training data from interactions with privacy preservation
        
        Args:
            interactions: List of user-item interaction records
            
        Returns:
            Tuple of (user_tensor, item_tensor, rating_tensor)
        """
        logger.info(f"Preparing data from {len(interactions)} interactions")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(interactions)
        
        # Apply differential privacy to ratings
        if 'rating' in df.columns:
            df['rating'] = self.dp_engine.add_laplace_noise(
                df['rating'].values,
                sensitivity=4.0,  # Rating scale is 1-5, so sensitivity is 4
                size=len(df)
            )
            # Clip ratings to valid range
            df['rating'] = np.clip(df['rating'], 1, 5)
        else:
            # For implicit feedback, create binary ratings
            df['rating'] = 1.0
        
        # Create user and item encoders
        unique_users = df['user_id'].unique()
        unique_items = df['app_id'].unique()
        
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_encoder = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item_id for item_id, idx in self.item_encoder.items()}
        
        # Encode users and items
        df['user_encoded'] = df['user_id'].map(self.user_encoder)
        df['item_encoded'] = df['app_id'].map(self.item_encoder)
        
        # Convert to tensors
        user_tensor = torch.LongTensor(df['user_encoded'].values)
        item_tensor = torch.LongTensor(df['item_encoded'].values)
        rating_tensor = torch.FloatTensor(df['rating'].values)
        
        logger.info(f"Data prepared: {len(unique_users)} users, {len(unique_items)} items")
        return user_tensor, item_tensor, rating_tensor
    
    async def train_model(
        self,
        interactions: List[Dict[str, Any]],
        validation_split: float = 0.2,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the collaborative filtering model
        
        Args:
            interactions: Training data
            validation_split: Fraction for validation
            save_path: Path to save the trained model
            
        Returns:
            Training metrics and history
        """
        logger.info("Starting collaborative filtering model training")
        
        # Prepare data
        user_tensor, item_tensor, rating_tensor = await self.prepare_data(interactions)
        
        # Split data
        indices = np.arange(len(user_tensor))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_split,
            random_state=42,
            stratify=None
        )
        
        train_users = user_tensor[train_idx]
        train_items = item_tensor[train_idx]
        train_ratings = rating_tensor[train_idx]
        
        val_users = user_tensor[val_idx]
        val_items = item_tensor[val_idx]
        val_ratings = rating_tensor[val_idx]
        
        # Initialize model
        self.model = NeuralCollaborativeFiltering(
            num_users=len(self.user_encoder),
            num_items=len(self.item_encoder),
            **{k: v for k, v in self.model_config.items() 
               if k in ['mf_dim', 'mlp_layers', 'dropout', 'activation']}
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        batch_size = self.model_config['batch_size']
        
        for epoch in range(self.model_config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_users), batch_size):
                batch_users = train_users[i:i+batch_size].to(self.device)
                batch_items = train_items[i:i+batch_size].to(self.device)
                batch_ratings = train_ratings[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items).squeeze()
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_users), batch_size):
                    batch_users = val_users[i:i+batch_size].to(self.device)
                    batch_items = val_items[i:i+batch_size].to(self.device)
                    batch_ratings = val_ratings[i:i+batch_size].to(self.device)
                    
                    predictions = self.model(batch_users, batch_items).squeeze()
                    loss = criterion(predictions, batch_ratings)
                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
            
            # Record history
            self.training_history['losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    await self._save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.model_config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        self.is_trained = True
        
        # Calculate final metrics
        final_metrics = await self._calculate_metrics(val_users, val_items, val_ratings)
        self.training_history['metrics'] = final_metrics
        
        logger.info("Training completed successfully")
        return {
            "training_history": self.training_history,
            "final_metrics": final_metrics,
            "model_config": self.model_config
        }
    
    async def _calculate_metrics(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        true_ratings: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                users.to(self.device),
                items.to(self.device)
            ).squeeze().cpu().numpy()
            
            true_ratings_np = true_ratings.numpy()
            
            mse = mean_squared_error(true_ratings_np, predictions)
            mae = mean_absolute_error(true_ratings_np, predictions)
            rmse = np.sqrt(mse)
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse)
            }
    
    async def predict(
        self,
        user_id: str,
        item_ids: List[str],
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for user-item pairs
        
        Args:
            user_id: User identifier
            item_ids: List of item identifiers
            return_scores: Whether to return prediction scores
            
        Returns:
            List of predictions with confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if user_id not in self.user_encoder:
            logger.warning(f"User {user_id} not in training data, using cold start strategy")
            return self._cold_start_predict(item_ids)
        
        user_idx = self.user_encoder[user_id]
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for item_id in item_ids:
                if item_id not in self.item_encoder:
                    continue
                
                item_idx = self.item_encoder[item_id]
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                
                predicted_rating = self.model(user_tensor, item_tensor).item()
                
                prediction_data = {
                    "item_id": item_id,
                    "predicted_rating": predicted_rating,
                    "confidence": min(1.0, predicted_rating / 5.0)
                }
                
                if return_scores:
                    prediction_data["raw_score"] = predicted_rating
                
                predictions.append(prediction_data)
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return predictions
    
    def _cold_start_predict(self, item_ids: List[str]) -> List[Dict[str, Any]]:
        """Handle cold start users with popularity-based recommendations"""
        # Simple popularity-based fallback
        return [
            {
                "item_id": item_id,
                "predicted_rating": 3.0,  # Average rating
                "confidence": 0.3,  # Low confidence for cold start
                "cold_start": True
            }
            for item_id in item_ids
        ]
    
    async def recommend(
        self,
        user_id: str,
        num_recommendations: int = 10,
        exclude_seen: bool = True,
        diversity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude already seen items
            diversity_threshold: Minimum diversity threshold
            
        Returns:
            List of recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating recommendations")
        
        # Get all available items
        all_items = list(self.item_encoder.keys())
        
        # Get predictions for all items
        predictions = await self.predict(user_id, all_items)
        
        # Filter out seen items if requested
        if exclude_seen:
            # This would need to query the database for user's interaction history
            # For now, we'll skip this filtering
            pass
        
        # Apply diversity filtering
        diverse_recommendations = self._apply_diversity_filter(
            predictions, diversity_threshold, num_recommendations
        )
        
        return diverse_recommendations[:num_recommendations]
    
    def _apply_diversity_filter(
        self,
        predictions: List[Dict[str, Any]],
        threshold: float,
        max_items: int
    ) -> List[Dict[str, Any]]:
        """Apply diversity filtering to recommendations"""
        if not predictions:
            return []
        
        selected = [predictions[0]]  # Start with highest rated
        
        for candidate in predictions[1:]:
            if len(selected) >= max_items:
                break
            
            # Simple diversity check - could be enhanced with semantic similarity
            is_diverse = True
            for selected_item in selected:
                # Placeholder diversity logic
                if candidate["item_id"] == selected_item["item_id"]:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(candidate)
        
        return selected
    
    async def _save_model(self, save_path: str):
        """Save trained model and encoders"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_decoder': self.user_decoder,
            'item_decoder': self.item_decoder,
            'model_config': self.model_config,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to {save_path}")
    
    async def load_model(self, load_path: str):
        """Load trained model and encoders"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        save_dict = torch.load(load_path, map_location=self.device)
        
        # Restore encoders
        self.user_encoder = save_dict['user_encoder']
        self.item_encoder = save_dict['item_encoder']
        self.user_decoder = save_dict['user_decoder']
        self.item_decoder = save_dict['item_decoder']
        self.model_config = save_dict['model_config']
        self.training_history = save_dict['training_history']
        
        # Initialize and load model
        self.model = NeuralCollaborativeFiltering(
            num_users=len(self.user_encoder),
            num_items=len(self.item_encoder),
            **{k: v for k, v in self.model_config.items() 
               if k in ['mf_dim', 'mlp_layers', 'dropout', 'activation']}
        ).to(self.device)
        
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.is_trained = True
        
        logger.info(f"Model loaded from {load_path}")

# Example usage and testing
if __name__ == "__main__":
    async def test_collaborative_filtering():
        # Sample interaction data
        sample_interactions = [
            {"user_id": "user1", "app_id": "app1", "rating": 5, "interaction_type": "rate"},
            {"user_id": "user1", "app_id": "app2", "rating": 3, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app1", "rating": 4, "interaction_type": "rate"},
            {"user_id": "user2", "app_id": "app3", "rating": 2, "interaction_type": "rate"},
            # Add more sample data...
        ]
        
        # Initialize and train model
        cf_engine = CollaborativeFilteringEngine()
        training_results = await cf_engine.train_model(sample_interactions)
        
        print("Training Results:", training_results)
        
        # Generate recommendations
        recommendations = await cf_engine.recommend("user1", num_recommendations=5)
        print("Recommendations for user1:", recommendations)
    
    # Run the test
    asyncio.run(test_collaborative_filtering())
