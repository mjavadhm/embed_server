# --- 1. Base Image ---
# Using a slim Python image to keep the size down.
FROM python:3.10-slim

# --- 2. Set Working Directory ---
WORKDIR /app

# --- 3. Caching Optimization: Install Dependencies First ---
# Copy ONLY the requirements file. This layer is cached as long as the file doesn't change.
COPY requirements.txt .

# Install dependencies using the specified index for PyTorch (CPU only).
RUN pip install --no-cache-dir -r requirements.txt

# --- 4. Copy Application Assets (Model and Database) ---
# These are large files, copied after dependency installation.
# The user must place the 'product_db' and 'embedding_model' directories in the build context.
COPY ./product_db /app/product_db
COPY ./embedding_model /app/embedding_model

# --- 5. Copy Application Code ---
# Finally, copy the rest of the application source code.
# Changes to the code will only invalidate this layer and subsequent layers.
COPY main.py .

# --- 6. Expose Port and Define Run Command ---
# Expose the port the FastAPI app will run on.
EXPOSE 8000

# Command to run the application using Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
