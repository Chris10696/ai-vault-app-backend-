#!/bin/bash

set -e

COMMAND=$1

function start_services() {
  echo "Starting AI Vault services..."

  # Build and start containers
  docker compose up -d --build

  # Wait for PostgreSQL to be healthy
  echo "Waiting for PostgreSQL to be healthy..."
  until docker exec ai-vault-postgres pg_isready -U postgres -d ai_vault; do
    echo -n "."
    sleep 2
  done
  echo "PostgreSQL is ready!"

  # Run database migrations
  echo "Applying database schema..."
  docker exec -i ai-vault-postgres psql -U postgres -d ai_vault < ../../scripts/setup/01_privacy_database_schema.sql

  # Wait for backend health check
  echo "Waiting for Backend API Gateway to be ready..."
  until curl -sf http://localhost:8000/health; do
    echo -n "."
    sleep 2
  done
  echo "Backend API Gateway is ready!"

  echo "All services started successfully!"
}

function stop_services() {
  echo "Stopping AI Vault services..."
  docker compose down
  echo "Services stopped."
}

case $COMMAND in
  start)
    start_services
    ;;
  stop)
    stop_services
    ;;
  restart)
    stop_services
    start_services
    ;;
  *)
    echo "Usage: $0 {start|stop|restart}"
    exit 1
    ;;
esac
