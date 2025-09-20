# --- 1. Use a slim and official Python base image ---
# This image is much smaller than the default one.
FROM python:3.9-slim-buster

# --- 2. Set the working directory inside the container ---
WORKDIR /app

# --- 3. Install the CPU-ONLY version of PyTorch ---
# This is the most important step to reduce image size.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# --- 4. Copy and install other Python dependencies ---
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- 5. Copy ONLY your application code ---
# !!! IMPORTANT: Change 'api.py' to the actual name of your Python file !!!
COPY ./main.py . 

# --- 6. Expose the port the app runs on ---
EXPOSE 8000

# --- 7. Define the command to run your application ---
# !!! IMPORTANT: Change 'api' to the name of your file (without .py) !!!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

