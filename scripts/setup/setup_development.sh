#!/bin/bash

# AI Vault - Module D Development Setup Script
set -e

echo "🚀 Setting up AI Vault Neural Discovery Engine (Module D)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.11"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "✅ Python $PYTHON_VERSION detected"
else
    echo "❌ Python $REQUIRED_VERSION or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create Python virtual environment
echo "📦 Creating Python virtual environment..."
cd packages/backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements_ml.txt

# Install development dependencies
pip install pytest pytest-asyncio httpx black isort flake8

echo "✅ Python environment setup complete"

# Go back to root directory
cd ../..

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:54322/aivault

# Redis Configuration
REDIS_URL=redis://localhost:6379

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-key-change-in-production-$(openssl rand -hex 32)
JWT_EXPIRATION_TIME_MINUTES=60

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=info

# ML Model Configuration
MODEL_STORAGE_PATH=./models
ENABLE_MODEL_TRAINING=true

# Privacy Configuration
DIFFERENTIAL_PRIVACY_EPSILON=0.5
DIFFERENTIAL_PRIVACY_DELTA=1e-5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=1

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8001
EOF
    echo "✅ .env file created with secure defaults"
else
    echo "ℹ️  .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models/{collaborative,content_based,hybrid}
mkdir -p logs
mkdir -p data/{raw,processed}
mkdir -p tests/reports

# Start services with Docker Compose
echo "🐳 Starting services with Docker Compose..."
docker-compose up -d db redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run database migrations
echo "🗄️  Running database migrations..."
cd packages/backend
source venv/bin/activate
python scripts/migrate/001_initial_setup.py

echo "✅ Database setup complete"

# Start the recommendation engine
echo "🤖 Starting Neural Discovery Engine..."
uvicorn services.discovery.main:app --host 0.0.0.0 --port 8001 --reload &
ENGINE_PID=$!

# Wait for the service to start
sleep 5

# Test the health endpoint
echo "🔍 Testing service health..."
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Neural Discovery Engine is running successfully!"
else
    echo "❌ Neural Discovery Engine failed to start"
    kill $ENGINE_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🎉 Setup complete! Your AI Vault Neural Discovery Engine is ready."
echo ""
echo "📋 Service Information:"
echo "   - API Server: http://localhost:8001"
echo "   - Health Check: http://localhost:8001/health"
echo "   - API Documentation: http://localhost:8001/docs"
echo "   - Database: localhost:54322"
echo "   - Redis: localhost:6379"
echo ""
echo "🚀 Next Steps:"
echo "   1. Train your first model: POST http://localhost:8001/api/v1/recommendations/train"
echo "   2. Check model status: GET http://localhost:8001/api/v1/recommendations/status"
echo "   3. Generate recommendations: POST http://localhost:8001/api/v1/recommendations/recommend"
echo ""
echo "📚 Documentation: Check the docs/ directory for detailed guides"
echo ""
echo "To stop the service: kill $ENGINE_PID"
