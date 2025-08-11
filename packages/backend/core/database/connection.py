import os
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:postgres@127.0.0.1:54322/postgres"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

@asynccontextmanager
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    async def create_tables():
        """Create all database tables"""
        from .models import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    
    @staticmethod
    async def drop_tables():
        """Drop all database tables"""
        from .models import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    
    @staticmethod
    async def check_connection():
        """Check database connection"""
        try:
            async with get_database() as db:
                result = await db.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

# Initialize database on startup
async def init_database():
    """Initialize database with required extensions and tables"""
    try:
        # Check if we can connect
        if not await DatabaseManager.check_connection():
            raise Exception("Cannot connect to database")
        
        # Create tables if they don't exist
        await DatabaseManager.create_tables()
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
