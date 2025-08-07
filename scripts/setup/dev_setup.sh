#!/bin/bash

# AI Vault Development Environment Setup Script
# Sets up local development environment with all dependencies

set -e

echo "🚀 Setting up AI Vault development environment..."

# Check if Python 3.11+ is installed
if ! command -v python3 &> /dev/null || [[ $(python3 -c 'import sys; print(sys.version_info >= (3, 11))') != "True" ]]; then
    echo "❌ Python 3.11+ is required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install pytest pytest-cov bandit safety black isort flake8

# Start Docker services
echo "🐳 Starting Docker services..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d redis postgres
elif command -v docker &> /dev/null; then
    docker run -d --name ai-vault-redis -p 6379:6379 redis:7-alpine
    docker run -d --name ai-vault-postgres -p 54322:5432 -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres supabase/postgres:15.1.0.147
else
    echo "⚠️  Docker not found. Please install Docker to run Redis and PostgreSQL."
fi

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Run database migrations
echo "🗄️  Setting up database schema..."
if command -v psql &> /dev/null; then
    PGPASSWORD=password psql -h localhost -p 54322 -U postgres -d postgres -f scripts/setup/database_schema.sql
else
    echo "⚠️  psql not found. Please run database_schema.sql manually."
fi

# Create environment file
echo "📝 Creating environment file..."
cp .env.example .env
echo "✏️  Please edit .env with your configuration"

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
pytest packages/backend/tests/unit/test_auth.py -v

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Start the API Gateway: cd packages/backend/services/api_gateway && python main.py"
echo "3. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "🔍 Useful commands:"
echo "  - Run tests: pytest"
echo "  - Security scan: python tools/security/zap_security_scan.py --target http://localhost:8000"
echo "  - Format code: black packages/ && isort packages/"
