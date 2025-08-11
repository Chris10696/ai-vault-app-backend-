from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, ARRAY, JSON, ForeignKey, BigInteger, DECIMAL
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from typing import Optional, Dict, Any, List

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    privacy_preferences = Column(JSONB, default={"allow_recommendations": True, "data_sharing": False})
    
    # Relationships
    interactions = relationship("UserInteraction", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user", uselist=False)

class App(Base):
    __tablename__ = "apps"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100), nullable=False)
    developer_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    tags = Column(ARRAY(String), default=[])
    features = Column(JSONB, default={})
    price_tier = Column(String(20), default="free")
    security_score = Column(Integer, default=0)
    content_embedding = Column(String)  # Will store vector as string for now
    feature_vector = Column(String)
    downloads_count = Column(BigInteger, default=0)
    rating_average = Column(DECIMAL(3, 2), default=0.0)
    rating_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    interactions = relationship("UserInteraction", back_populates="app")

class UserInteraction(Base):
    __tablename__ = "user_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    app_id = Column(UUID(as_uuid=True), ForeignKey("apps.id", ondelete="CASCADE"))
    interaction_type = Column(String(50), nullable=False)
    rating = Column(Integer)
    interaction_metadata = Column(JSONB, default={})
    interaction_time = Column(DateTime(timezone=True), server_default=func.now())
    session_id = Column(UUID(as_uuid=True))
    context_data = Column(JSONB, default={})
    privacy_level = Column(Integer, default=3)
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    app = relationship("App", back_populates="interactions")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    category_preferences = Column(JSONB, default={})
    feature_preferences = Column(JSONB, default={})
    recommendation_settings = Column(JSONB, default={
        "enable_collaborative": True,
        "enable_content_based": True,
        "novelty_preference": 0.5,
        "diversity_preference": 0.7
    })
    privacy_budget = Column(DECIMAL(10, 6), default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="preferences")

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    model_architecture = Column(Text)
    hyperparameters = Column(JSONB)
    performance_metrics = Column(JSONB)
    model_path = Column(Text)
    training_data_hash = Column(String(64))
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    trained_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))

class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml_models.id"))
    recommendations = Column(JSONB, nullable=False)
    recommendation_context = Column(JSONB, default={})
    model_confidence = Column(DECIMAL(5, 4))
    explanation = Column(JSONB)
    served_at = Column(DateTime(timezone=True), server_default=func.now())
    user_feedback = Column(Integer)
    feedback_time = Column(DateTime(timezone=True))
    ab_test_group = Column(String(10))
    session_id = Column(UUID(as_uuid=True))

class ItemSimilarity(Base):
    __tablename__ = "item_similarities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_a_id = Column(UUID(as_uuid=True), ForeignKey("apps.id", ondelete="CASCADE"))
    item_b_id = Column(UUID(as_uuid=True), ForeignKey("apps.id", ondelete="CASCADE"))
    similarity_score = Column(DECIMAL(8, 6), nullable=False)
    similarity_type = Column(String(50), nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    model_version = Column(String(20))
