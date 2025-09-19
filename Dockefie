# Stage 1: Build Stage
# Use a specific Python version for reproducibility
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy dependency definition files
COPY requirements.txt .

# Install dependencies into a virtual environment
# This caches the dependencies in a separate layer
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Stage
# Use the same base image for a smaller final image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY main.py .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# --host 0.0.0.0 is crucial for it to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

