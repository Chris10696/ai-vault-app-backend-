-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "extensions";

-- Users table (if not already exists from previous modules)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    privacy_preferences JSONB DEFAULT '{"allow_recommendations": true, "data_sharing": false}'::jsonb
);

-- Apps table for recommendation system
CREATE TABLE apps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    developer_id UUID REFERENCES users(id) ON DELETE CASCADE,
    tags TEXT[] DEFAULT '{}',
    features JSONB DEFAULT '{}',
    price_tier VARCHAR(20) DEFAULT 'free',
    security_score INTEGER DEFAULT 0,
    content_embedding VECTOR(1536), -- OpenAI embeddings dimension
    feature_vector VECTOR(512), -- Custom feature embeddings
    downloads_count BIGINT DEFAULT 0,
    rating_average DECIMAL(3,2) DEFAULT 0.0,
    rating_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- User interactions for collaborative filtering
CREATE TABLE user_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    app_id UUID REFERENCES apps(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL, -- 'view', 'download', 'rate', 'favorite'
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    interaction_metadata JSONB DEFAULT '{}',
    interaction_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id UUID,
    context_data JSONB DEFAULT '{}', -- Device, location, time context
    privacy_level INTEGER DEFAULT 3 -- 1=public, 2=friends, 3=private
);

-- User preferences for personalization
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    category_preferences JSONB DEFAULT '{}',
    feature_preferences JSONB DEFAULT '{}',
    recommendation_settings JSONB DEFAULT '{"enable_collaborative": true, "enable_content_based": true, "novelty_preference": 0.5, "diversity_preference": 0.7}'::jsonb,
    privacy_budget DECIMAL(10,6) DEFAULT 1.0, -- Differential privacy budget
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML model metadata and versioning
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'collaborative', 'content_based', 'hybrid'
    model_version VARCHAR(20) NOT NULL,
    model_architecture TEXT,
    hyperparameters JSONB,
    performance_metrics JSONB,
    model_path TEXT,
    training_data_hash VARCHAR(64),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    trained_by UUID REFERENCES users(id)
);

-- Training datasets tracking
CREATE TABLE training_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(100) NOT NULL,
    dataset_version VARCHAR(20) DEFAULT '1.0',
    data_source VARCHAR(100),
    data_schema JSONB,
    privacy_level INTEGER DEFAULT 3,
    epsilon_value DECIMAL(10,6), -- For differential privacy
    record_count BIGINT,
    feature_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_hash VARCHAR(64) UNIQUE
);

-- Recommendation logs for A/B testing and monitoring
CREATE TABLE recommendation_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    model_id UUID REFERENCES ml_models(id),
    recommendations JSONB NOT NULL,
    recommendation_context JSONB DEFAULT '{}',
    model_confidence DECIMAL(5,4),
    explanation JSONB, -- For explainable recommendations
    served_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_feedback INTEGER, -- 1-5 rating
    feedback_time TIMESTAMP WITH TIME ZONE,
    ab_test_group VARCHAR(10), -- A, B, control
    session_id UUID
);

-- Similarity matrices for collaborative filtering
CREATE TABLE item_similarities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    item_a_id UUID REFERENCES apps(id) ON DELETE CASCADE,
    item_b_id UUID REFERENCES apps(id) ON DELETE CASCADE,
    similarity_score DECIMAL(8,6) NOT NULL,
    similarity_type VARCHAR(50) NOT NULL, -- 'cosine', 'pearson', 'jaccard'
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(20),
    UNIQUE(item_a_id, item_b_id, similarity_type)
);

-- User similarities for collaborative filtering
CREATE TABLE user_similarities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_a_id UUID REFERENCES users(id) ON DELETE CASCADE,
    user_b_id UUID REFERENCES users(id) ON DELETE CASCADE,
    similarity_score DECIMAL(8,6) NOT NULL,
    similarity_type VARCHAR(50) NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(20),
    privacy_preserved BOOLEAN DEFAULT TRUE,
    UNIQUE(user_a_id, user_b_id, similarity_type)
);

-- Indexes for performance optimization
CREATE INDEX idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX idx_user_interactions_app_id ON user_interactions(app_id);
CREATE INDEX idx_user_interactions_type ON user_interactions(interaction_type);
CREATE INDEX idx_user_interactions_time ON user_interactions(interaction_time);
CREATE INDEX idx_apps_category ON apps(category);
CREATE INDEX idx_apps_rating ON apps(rating_average);
CREATE INDEX idx_recommendation_logs_user_id ON recommendation_logs(user_id);
CREATE INDEX idx_recommendation_logs_served_at ON recommendation_logs(served_at);

-- Vector similarity search indexes
CREATE INDEX idx_apps_content_embedding ON apps USING hnsw (content_embedding vector_cosine_ops);
CREATE INDEX idx_apps_feature_embedding ON apps USING hnsw (feature_vector vector_cosine_ops);
