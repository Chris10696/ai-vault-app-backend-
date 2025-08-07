-- Privacy module database schema
-- Run this after the main database_schema.sql

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User privacy profiles table
CREATE TABLE IF NOT EXISTS user_privacy_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    preferred_privacy_level VARCHAR(20) DEFAULT 'standard' CHECK (preferred_privacy_level IN ('minimal', 'standard', 'high', 'maximum')),
    total_epsilon_budget DECIMAL(10,6) DEFAULT 10.0 CHECK (total_epsilon_budget > 0),
    daily_epsilon_limit DECIMAL(10,6) DEFAULT 2.0 CHECK (daily_epsilon_limit > 0),
    epsilon_used_today DECIMAL(10,6) DEFAULT 0.0 CHECK (epsilon_used_today >= 0),
    epsilon_used_total DECIMAL(10,6) DEFAULT 0.0 CHECK (epsilon_used_total >= 0),
    last_reset_date TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Privacy audit logs table
CREATE TABLE IF NOT EXISTS privacy_audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    operation_type VARCHAR(50) NOT NULL,
    query_id UUID,
    epsilon_consumed DECIMAL(10,6) NOT NULL CHECK (epsilon_consumed >= 0),
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Privacy queries table (for tracking and analytics)
CREATE TABLE IF NOT EXISTS privacy_queries (
    query_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    query_type VARCHAR(20) NOT NULL CHECK (query_type IN ('count', 'sum', 'mean', 'median', 'histogram', 'range_query')),
    dataset_id VARCHAR(100) NOT NULL,
    query_params JSONB DEFAULT '{}',
    privacy_level VARCHAR(20) DEFAULT 'standard',
    epsilon_used DECIMAL(10,6) NOT NULL,
    noise_magnitude DECIMAL(10,6),
    accuracy_loss DECIMAL(10,6),
    execution_time_ms INTEGER,
    requested_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed', 'budget_exceeded'))
);

-- Dataset metadata table (for managing query sensitivities)
CREATE TABLE IF NOT EXISTS dataset_metadata (
    dataset_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    record_count INTEGER DEFAULT 0,
    columns JSONB DEFAULT '[]',
    data_types JSONB DEFAULT '{}',
    sensitivity_levels JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Encrypted user vault table
CREATE TABLE IF NOT EXISTS encrypted_user_vault (
    entry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    encrypted_data TEXT NOT NULL, -- Base64 encoded encrypted JSON
    data_type VARCHAR(50) DEFAULT 'generic',
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_privacy_profiles_user_id ON user_privacy_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_privacy_audit_user_timestamp ON privacy_audit_logs(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_privacy_queries_user_status ON privacy_queries(user_id, status);
CREATE INDEX IF NOT EXISTS idx_privacy_queries_timestamp ON privacy_queries(requested_at DESC);
CREATE INDEX IF NOT EXISTS idx_vault_user_id ON encrypted_user_vault(user_id);
CREATE INDEX IF NOT EXISTS idx_vault_created_at ON encrypted_user_vault(created_at DESC);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_privacy_profiles_updated_at BEFORE UPDATE ON user_privacy_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dataset_metadata_updated_at BEFORE UPDATE ON dataset_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_vault_updated_at BEFORE UPDATE ON encrypted_user_vault FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to reset daily privacy budgets (called by cron job)
CREATE OR REPLACE FUNCTION reset_daily_privacy_budgets()
RETURNS INTEGER AS $$
DECLARE
    reset_count INTEGER := 0;
BEGIN
    UPDATE user_privacy_profiles 
    SET 
        epsilon_used_today = 0.0,
        last_reset_date = NOW(),
        updated_at = NOW()
    WHERE 
        DATE(last_reset_date) < CURRENT_DATE;
    
    GET DIAGNOSTICS reset_count = ROW_COUNT;
    
    -- Log the reset operation
    INSERT INTO privacy_audit_logs (user_id, operation_type, epsilon_consumed, metadata)
    SELECT 
        user_id, 
        'daily_budget_reset', 
        0.0,
        jsonb_build_object('reset_count', reset_count, 'reset_date', CURRENT_DATE)
    FROM user_privacy_profiles 
    WHERE DATE(last_reset_date) = CURRENT_DATE
    LIMIT 1;
    
    RETURN reset_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get privacy budget status
CREATE OR REPLACE FUNCTION get_privacy_budget_status(p_user_id UUID)
RETURNS TABLE (
    daily_limit DECIMAL(10,6),
    daily_used DECIMAL(10,6),
    daily_remaining DECIMAL(10,6),
    total_limit DECIMAL(10,6),
    total_used DECIMAL(10,6),
    total_remaining DECIMAL(10,6),
    last_reset DATE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        upp.daily_epsilon_limit,
        upp.epsilon_used_today,
        (upp.daily_epsilon_limit - upp.epsilon_used_today) as daily_remaining,
        upp.total_epsilon_budget,
        upp.epsilon_used_total,
        (upp.total_epsilon_budget - upp.epsilon_used_total) as total_remaining,
        DATE(upp.last_reset_date)
    FROM user_privacy_profiles upp
    WHERE upp.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- Function to safely consume privacy budget
CREATE OR REPLACE FUNCTION consume_privacy_budget(
    p_user_id UUID,
    p_epsilon_amount DECIMAL(10,6),
    p_query_id UUID DEFAULT NULL
)
RETURNS TABLE (
    success BOOLEAN,
    message TEXT,
    remaining_daily DECIMAL(10,6),
    remaining_total DECIMAL(10,6)
) AS $$
DECLARE
    current_profile user_privacy_profiles%ROWTYPE;
    new_daily_total DECIMAL(10,6);
    new_total_used DECIMAL(10,6);
BEGIN
    -- Get current profile with row lock
    SELECT * INTO current_profile
    FROM user_privacy_profiles
    WHERE user_id = p_user_id
    FOR UPDATE;
    
    -- Check if user exists
    IF NOT FOUND THEN
        -- Create default profile
        INSERT INTO user_privacy_profiles (user_id)
        VALUES (p_user_id)
        RETURNING * INTO current_profile;
    END IF;
    
    -- Reset daily budget if needed
    IF DATE(current_profile.last_reset_date) < CURRENT_DATE THEN
        UPDATE user_privacy_profiles
        SET 
            epsilon_used_today = 0.0,
            last_reset_date = NOW(),
            updated_at = NOW()
        WHERE user_id = p_user_id;
        
        current_profile.epsilon_used_today := 0.0;
    END IF;
    
    -- Calculate new totals
    new_daily_total := current_profile.epsilon_used_today + p_epsilon_amount;
    new_total_used := current_profile.epsilon_used_total + p_epsilon_amount;
    
    -- Check daily budget
    IF new_daily_total > current_profile.daily_epsilon_limit THEN
        RETURN QUERY SELECT 
            FALSE, 
            'Daily privacy budget exceeded',
            (current_profile.daily_epsilon_limit - current_profile.epsilon_used_today),
            (current_profile.total_epsilon_budget - current_profile.epsilon_used_total);
        RETURN;
    END IF;
    
    -- Check total budget
    IF new_total_used > current_profile.total_epsilon_budget THEN
        RETURN QUERY SELECT 
            FALSE, 
            'Total privacy budget exceeded',
            (current_profile.daily_epsilon_limit - current_profile.epsilon_used_today),
            (current_profile.total_epsilon_budget - current_profile.epsilon_used_total);
        RETURN;
    END IF;
    
    -- Consume budget
    UPDATE user_privacy_profiles
    SET 
        epsilon_used_today = new_daily_total,
        epsilon_used_total = new_total_used,
        updated_at = NOW()
    WHERE user_id = p_user_id;
    
    -- Create audit log
    INSERT INTO privacy_audit_logs (user_id, operation_type, query_id, epsilon_consumed)
    VALUES (p_user_id, 'budget_consumed', p_query_id, p_epsilon_amount);
    
    -- Return success
    RETURN QUERY SELECT 
        TRUE, 
        'Budget consumed successfully',
        (current_profile.daily_epsilon_limit - new_daily_total),
        (current_profile.total_epsilon_budget - new_total_used);
END;
$$ LANGUAGE plpgsql;

-- Sample dataset metadata
INSERT INTO dataset_metadata (dataset_id, name, description, record_count, columns, data_types, sensitivity_levels) VALUES
('demo_sales', 'Sales Dataset', 'Sample sales data for testing', 10000, 
 '["customer_id", "product_id", "amount", "date", "region"]'::jsonb,
 '{"customer_id": "integer", "product_id": "integer", "amount": "decimal", "date": "timestamp", "region": "string"}'::jsonb,
 '{"customer_id": 1.0, "product_id": 1.0, "amount": 100.0, "date": 1.0, "region": 1.0}'::jsonb),
 
('demo_users', 'User Dataset', 'Sample user data for testing', 5000,
 '["user_id", "age", "salary", "department", "join_date"]'::jsonb,
 '{"user_id": "integer", "age": "integer", "salary": "decimal", "department": "string", "join_date": "timestamp"}'::jsonb,
 '{"user_id": 1.0, "age": 5.0, "salary": 10000.0, "department": 1.0, "join_date": 1.0}'::jsonb);

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;
