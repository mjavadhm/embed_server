# --- 1. Base Image ---
FROM python:3.10-slim

# --- 2. Set Working Directory ---
WORKDIR /app

# --- 3. Create Mount Point for All Persistent Data ---
# این دایرکتوری برای ذخیره دیتابیس و مدل امبدینگ استفاده خواهد شد
RUN mkdir -p /app/product_db

# --- 4. Install Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 5. Copy Application Code ---
COPY main.py .

# --- 6. Expose Port and Define Run Command ---
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
