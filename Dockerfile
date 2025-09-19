# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency definition file
COPY requirements.txt .

# Install dependencies. The venv is not strictly necessary in a container
# but we install gdown which requires pip.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY main.py .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# This is the standard way and is less likely to be overridden by platforms.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

