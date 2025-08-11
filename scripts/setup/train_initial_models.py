"""
Script to train initial ML models for the Neural Discovery Engine
"""
import asyncio
import httpx
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8001/api/v1/recommendations"
ADMIN_TOKEN = "admin_token_here"  # In production, use proper authentication

async def train_models():
    """Train all recommendation models"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        
        # Train collaborative filtering model
        logger.info("Starting collaborative filtering model training...")
        response = await client.post(
            f"{API_BASE_URL}/train",
            json={
                "model_type": "collaborative",
                "validation_split": 0.2,
                "force_retrain": False
            },
            headers=headers
        )
        
        if response.status_code == 200:
            cf_training_id = response.json()["training_id"]
            logger.info(f"Collaborative training started: {cf_training_id}")
        else:
            logger.error(f"Failed to start collaborative training: {response.text}")
            return
        
        # Train content-based filtering model
        logger.info("Starting content-based filtering model training...")
        response = await client.post(
            f"{API_BASE_URL}/train",
            json={
                "model_type": "content_based",
                "force_retrain": False
            },
            headers=headers
        )
        
        if response.status_code == 200:
            cb_training_id = response.json()["training_id"]
            logger.info(f"Content-based training started: {cb_training_id}")
        else:
            logger.error(f"Failed to start content-based training: {response.text}")
            return
        
        # Wait for training to complete
        logger.info("Waiting for training to complete...")
        await asyncio.sleep(60)  # Wait 1 minute
        
        # Train hybrid model
        logger.info("Starting hybrid model training...")
        response = await client.post(
            f"{API_BASE_URL}/train",
            json={
                "model_type": "hybrid",
                "validation_split": 0.2,
                "force_retrain": False
            },
            headers=headers
        )
        
        if response.status_code == 200:
            hybrid_training_id = response.json()["training_id"]
            logger.info(f"Hybrid training started: {hybrid_training_id}")
        else:
            logger.error(f"Failed to start hybrid training: {response.text}")
            return
        
        # Check final model status
        logger.info("Checking final model status...")
        response = await client.get(f"{API_BASE_URL}/status")
        
        if response.status_code == 200:
            status = response.json()
            logger.info(f"Final model status: {json.dumps(status, indent=2)}")
            
            if status["hybrid_ready"]:
                logger.info("✅ All models trained successfully!")
            else:
                logger.warning("⚠️  Some models may still be training")
        else:
            logger.error(f"Failed to get model status: {response.text}")

if __name__ == "__main__":
    asyncio.run(train_models())
