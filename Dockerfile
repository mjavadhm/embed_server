# --- 1. Use a newer, slim, and official Python base image ---
# "bookworm" has a newer version of sqlite3 required by chromadb
FROM python:3.9-slim-bookworm

# --- 2. Set the working directory inside the container ---
WORKDIR /app

# --- 3. Copy ONLY the requirements file first ---
# این مهم‌ترین مرحله برای استفاده از کش است
COPY requirements.txt ./

# --- 4. Install all dependencies in a single layer ---
# تا زمانی که فایل requirements.txt تغییر نکند، این لایه از کش خوانده می‌شود
# و نیازی به دانلود و نصب مجدد پکیج‌ها نیست
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- 5. Copy your application code LAST ---
# چون کد برنامه بیشتر از نیازمندی‌ها تغییر می‌کند، آن را در انتها کپی می‌کنیم
# تا تغییرات آن باعث بی‌اعتبار شدن لایه نصب پکیج‌ها نشود
COPY ./main.py .

# --- 6. Expose the port the app runs on ---
EXPOSE 8000

# --- 7. Define the command to run your application ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
