"""
Initial database setup and migration
"""
import asyncio
import asyncpg
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")

async def run_migration():
    """Run the initial database migration"""
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Read and execute the schema file
        schema_file = Path(__file__).parent.parent.parent / "core" / "database" / "recommendation_schema.sql"
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema creation
        await conn.execute(schema_sql)
        
        logger.info("Database migration completed successfully")
        
        # Insert some sample data for testing
        await insert_sample_data(conn)
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

async def insert_sample_data(conn):
    """Insert sample data for testing"""
    try:
        # Insert sample users
        await conn.execute("""
            INSERT INTO users (email, username, privacy_preferences) VALUES
            ('admin@aivault.com', 'admin', '{"allow_recommendations": true, "data_sharing": false}'),
            ('user1@test.com', 'testuser1', '{"allow_recommendations": true, "data_sharing": true}'),
            ('user2@test.com', 'testuser2', '{"allow_recommendations": true, "data_sharing": false}')
            ON CONFLICT (email) DO NOTHING;
        """)
        
        # Insert sample apps
        await conn.execute("""
            INSERT INTO apps (name, description, category, tags, features, price_tier, security_score, rating_average, rating_count, downloads_count) VALUES
            ('Photo Editor Pro', 'Advanced photo editing with AI filters', 'Photography', ARRAY['photo', 'editor', 'ai'], '{"filters": 50, "ai_tools": 10}', 'premium', 95, 4.5, 1250, 50000),
            ('Task Manager', 'Organize your daily tasks efficiently', 'Productivity', ARRAY['productivity', 'tasks', 'organization'], '{"templates": 20, "collaboration": true}', 'free', 90, 4.2, 850, 25000),
            ('Video Player', 'High-quality video playback', 'Entertainment', ARRAY['video', 'media', 'player'], '{"formats": 15, "quality": "4K"}', 'freemium', 88, 4.0, 600, 75000),
            ('Weather App', 'Accurate weather forecasting', 'Utilities', ARRAY['weather', 'forecast', 'location'], '{"locations": 100, "alerts": true}', 'free', 92, 4.3, 920, 100000),
            ('Fitness Tracker', 'Track your fitness goals', 'Health', ARRAY['fitness', 'health', 'tracking'], '{"workouts": 200, "nutrition": true}', 'premium', 94, 4.4, 1100, 30000)
            ON CONFLICT (name) DO NOTHING;
        """)
        
        # Get user and app IDs for sample interactions
        users = await conn.fetch("SELECT id FROM users LIMIT 3")
        apps = await conn.fetch("SELECT id FROM apps LIMIT 5")
        
        if users and apps:
            # Insert sample interactions
            interactions = []
            import random
            
            for user in users:
                for app in apps:
                    if random.random() > 0.6:  # 40% interaction rate
                        rating = random.randint(1, 5)
                        interaction_type = random.choice(['rate', 'view', 'download'])
                        interactions.append((user['id'], app['id'], interaction_type, rating))
            
            if interactions:
                await conn.executemany("""
                    INSERT INTO user_interactions (user_id, app_id, interaction_type, rating)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                """, interactions)
        
        logger.info("Sample data inserted successfully")
        
    except Exception as e:
        logger.warning(f"Failed to insert sample data: {e}")

if __name__ == "__main__":
    asyncio.run(run_migration())
