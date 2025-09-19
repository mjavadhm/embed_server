FROM python:3.10-slim

# Set working directory
WORKDIR /app

# --- Caching Optimization ---

# 1. Copy ONLY the requirements file first.
# This file changes infrequently.
COPY requirements.txt .

# 2. Install dependencies.
# This layer will now be cached as long as requirements.txt doesn't change.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 3. NOW, copy the rest of the application code.
# If only main.py changes, the cache up to the previous step remains valid,
# and pip install will NOT run again.
COPY main.py .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

