# --- مرحله ۱: بیلد (Builder Stage) ---
FROM python:3.10-alpine as builder

# ✨ متغیر برای پاک کردن کش. با تغییر این عدد، بیلد جدید اجباری می‌شود. ✨
ENV CACHE_BUSTER=1

# نصب تمام ابزارهای کامپایل مورد نیاز
# build-base: ابزارهای اصلی کامپایل
# libffi-dev, openssl-dev: برای برخی کتابخانه‌های رمزنگاری و شبکه
RUN apk add --no-cache build-base g++ libffi-dev openssl-dev

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- مرحله ۲: نهایی (Final Stage) ---
FROM python:3.10-alpine

# فقط کتابخانه‌های زمان اجرا را نصب می‌کنیم
RUN apk add --no-cache libgomp libffi

WORKDIR /app

RUN mkdir -p /app/product_db

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
