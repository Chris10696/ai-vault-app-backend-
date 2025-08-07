# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy backend source code
COPY ./packages/backend ./packages/backend

# Install system dependencies for psycopg2 and others
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose backend port
EXPOSE 8000

# Start command
CMD ["uvicorn", "packages.backend.services.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
