-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    pwd_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- User sessions table for refresh token management
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    fp_hash CHAR(64) NOT NULL,
    refresh_sig CHAR(64) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    last_seen TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for session management
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions() RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM user_sessions WHERE expires_at < NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically clean up expired sessions
DROP TRIGGER IF EXISTS trigger_cleanup_sessions ON user_sessions;
CREATE TRIGGER trigger_cleanup_sessions
    AFTER INSERT ON user_sessions
    EXECUTE FUNCTION cleanup_expired_sessions();

-- Zero trust policies table (for Sprint A2)
CREATE TABLE IF NOT EXISTS zero_trust_policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_pattern TEXT NOT NULL,
    required_trust_level SMALLINT CHECK (required_trust_level BETWEEN 0 AND 100),
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Security events table (for Sprint A3)
CREATE TABLE IF NOT EXISTS security_events (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    ip INET,
    event_type TEXT NOT NULL,
    meta JSONB DEFAULT '{}',
    severity TEXT DEFAULT 'medium',
    ts TIMESTAMP DEFAULT NOW()
);

-- Create indexes for security events
CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_event_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_ts ON security_events(ts);

-- Insert default zero trust policies
INSERT INTO zero_trust_policies (resource_pattern, required_trust_level, conditions) VALUES
('/api/apps/*', 40, '{"methods": ["GET"], "description": "View apps"}'),
('/api/apps/create', 70, '{"methods": ["POST"], "description": "Create apps"}'),
('/api/vault/*', 60, '{"methods": ["GET", "POST"], "description": "Access vault"}'),
('/api/memory/*', 80, '{"methods": ["GET", "POST", "PUT", "DELETE"], "description": "Memory operations"}')
ON CONFLICT DO NOTHING;
