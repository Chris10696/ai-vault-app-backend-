-- Database initialization script for AI Vault Backend
-- Includes tables for homomorphic encryption key management and model storage

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Zero Trust Policies
CREATE TABLE zero_trust_policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_pattern VARCHAR(255) NOT NULL,
    required_trust_level INTEGER NOT NULL,
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- App Registry (Blockchain Simulation)
CREATE TABLE app_registry (
    app_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    developer_id UUID NOT NULL,
    app_hash VARCHAR(64) NOT NULL UNIQUE,
    security_score INTEGER DEFAULT 0,
    verification_status BOOLEAN DEFAULT FALSE,
    audit_trail JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Homomorphic Encryption Keys
CREATE TABLE he_encryption_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_id VARCHAR(255) NOT NULL UNIQUE,
    key_type VARCHAR(50) NOT NULL,
    context_name VARCHAR(100) NOT NULL,
    security_level INTEGER NOT NULL,
    encrypted_key_data BYTEA NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'ACTIVE',
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    CONSTRAINT valid_key_type CHECK (key_type IN ('seal_secret', 'seal_public', 'galois', 'relinearization', 'bootstrap', 'master')),
    CONSTRAINT valid_status CHECK (status IN ('ACTIVE', 'EXPIRING', 'ARCHIVED', 'REVOKED'))
);

-- Encrypted Models
CREATE TABLE he_encrypted_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,
    encryption_context VARCHAR(100) NOT NULL,
    model_metadata JSONB NOT NULL DEFAULT '{}',
    architecture JSONB NOT NULL DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    security_level INTEGER NOT NULL,
    file_path VARCHAR(500),
    file_size BIGINT,
    checksum VARCHAR(64),
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_by UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_model_status CHECK (status IN ('ACTIVE', 'INACTIVE', 'TRAINING', 'DEPRECATED'))
);

-- Inference Sessions
CREATE TABLE he_inference_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL UNIQUE,
    model_id VARCHAR(255) NOT NULL,
    encryption_context VARCHAR(100) NOT NULL,
    user_id UUID,
    request_count INTEGER DEFAULT 0,
    total_inference_time FLOAT DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours'),
    
    CONSTRAINT valid_session_status CHECK (status IN ('ACTIVE', 'INACTIVE', 'EXPIRED')),
    CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES he_encrypted_models(model_id) ON DELETE CASCADE
);

-- Performance Metrics
CREATE TABLE he_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation_type VARCHAR(100) NOT NULL,
    execution_time FLOAT NOT NULL,
    memory_usage FLOAT DEFAULT 0.0,
    cpu_usage FLOAT DEFAULT 0.0,
    input_size INTEGER DEFAULT 0,
    context_name VARCHAR(100),
    optimization_applied TEXT[] DEFAULT '{}',
    session_id VARCHAR(255),
    model_id VARCHAR(255),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_session_id FOREIGN KEY (session_id) REFERENCES he_inference_sessions(session_id) ON DELETE SET NULL
);

-- Pipeline Executions
CREATE TABLE he_pipeline_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id VARCHAR(255) NOT NULL UNIQUE,
    config_id VARCHAR(255) NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    current_stage VARCHAR(50) NOT NULL,
    stage_timings JSONB DEFAULT '{}',
    total_time FLOAT DEFAULT 0.0,
    input_size INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    results_cache_key VARCHAR(255),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    CONSTRAINT valid_pipeline_stage CHECK (current_stage IN ('validation', 'preprocessing', 'encryption', 'inference', 'postprocessing', 'decryption', 'complete'))
);

-- Key Audit Log
CREATE TABLE he_key_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation VARCHAR(50) NOT NULL,
    key_id VARCHAR(255) NOT NULL,
    user_id UUID,
    trust_score FLOAT,
    details TEXT,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_key_audit_key_id FOREIGN KEY (key_id) REFERENCES he_encryption_keys(key_id) ON DELETE CASCADE
);

-- Security Events
CREATE TABLE he_security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    user_id UUID,
    description TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_severity CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

-- Create indexes for performance
CREATE INDEX idx_he_keys_key_id ON he_encryption_keys(key_id);
CREATE INDEX idx_he_keys_type_context ON he_encryption_keys(key_type, context_name);
CREATE INDEX idx_he_keys_status ON he_encryption_keys(status);
CREATE INDEX idx_he_keys_created_at ON he_encryption_keys(created_at);

CREATE INDEX idx_he_models_model_id ON he_encrypted_models(model_id);
CREATE INDEX idx_he_models_type ON he_encrypted_models(model_type);
CREATE INDEX idx_he_models_status ON he_encrypted_models(status);
CREATE INDEX idx_he_models_created_at ON he_encrypted_models(created_at);

CREATE INDEX idx_he_sessions_session_id ON he_inference_sessions(session_id);
CREATE INDEX idx_he_sessions_model_id ON he_inference_sessions(model_id);
CREATE INDEX idx_he_sessions_status ON he_inference_sessions(status);
CREATE INDEX idx_he_sessions_expires_at ON he_inference_sessions(expires_at);

CREATE INDEX idx_he_metrics_operation_type ON he_performance_metrics(operation_type);
CREATE INDEX idx_he_metrics_timestamp ON he_performance_metrics(timestamp);
CREATE INDEX idx_he_metrics_session_id ON he_performance_metrics(session_id);

CREATE INDEX idx_he_executions_execution_id ON he_pipeline_executions(execution_id);
CREATE INDEX idx_he_executions_model_id ON he_pipeline_executions(model_id);
CREATE INDEX idx_he_executions_started_at ON he_pipeline_executions(started_at);

CREATE INDEX idx_he_audit_key_id ON he_key_audit_log(key_id);
CREATE INDEX idx_he_audit_timestamp ON he_key_audit_log(timestamp);
CREATE INDEX idx_he_audit_operation ON he_key_audit_log(operation);

CREATE INDEX idx_he_security_event_type ON he_security_events(event_type);
CREATE INDEX idx_he_security_severity ON he_security_events(severity);
CREATE INDEX idx_he_security_resolved ON he_security_events(resolved);
CREATE INDEX idx_he_security_created_at ON he_security_events(created_at);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_zero_trust_policies_updated_at BEFORE UPDATE ON zero_trust_policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_app_registry_updated_at BEFORE UPDATE ON app_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_he_encrypted_models_updated_at BEFORE UPDATE ON he_encrypted_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default zero trust policies for HE operations
INSERT INTO zero_trust_policies (resource_pattern, required_trust_level, conditions) VALUES
('he:encrypt:*', 60, '{"max_operations_per_hour": 1000}'),
('he:decrypt:*', 70, '{"max_operations_per_hour": 500}'),
('he:model:upload', 80, '{"require_mfa": true, "max_model_size_mb": 100}'),
('he:model:execute', 60, '{"max_concurrent_sessions": 10}'),
('he:key:generate', 90, '{"require_admin": true, "audit_required": true}'),
('he:key:access', 70, '{"max_failed_attempts": 3, "lockout_duration_minutes": 15}');

-- Insert sample app registry entry
INSERT INTO app_registry (developer_id, app_hash, security_score, verification_status, audit_trail) VALUES
(uuid_generate_v4(), 'sample_app_hash_12345', 85, TRUE, '[{"action": "created", "timestamp": "2024-01-01T00:00:00Z", "by": "system"}]');

COMMIT;
