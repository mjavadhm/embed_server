# Stage 1: Build Stage
# Use a specific Python version for reproducibility
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies that might be needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy dependency definition files
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# gdown is now in requirements.txt and will be installed here
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Stage
# Use the same base image
FROM python:3.10-slim

# Install runtime system dependencies (unzip for the startup script)
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code and the new startup script
COPY main.py .
COPY startup.sh .

# Make the startup script executable
RUN chmod +x ./startup.sh

# Expose the port the app will run on
EXPOSE 8000

# Use the startup script as the command to run the container.
# This script will handle DB download and then start the Uvicorn server.
CMD ["./startup.sh"]

