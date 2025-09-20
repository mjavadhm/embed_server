# --- مرحله ۱: بیلد (Builder Stage) ---
# از ایمیج کامل Alpine برای نصب استفاده می‌کنیم که ابزارهای کامپایل را دارد
FROM python:3.10-alpine as builder

# نصب ابزارهای مورد نیاز برای کامپایل پکیج‌هایی مانند numpy
RUN apk add --no-cache build-base

WORKDIR /app

# کپی و نصب نیازمندی‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- مرحله ۲: نهایی (Final Stage) ---
# از ایمیج بسیار سبک Alpine برای اجرای برنامه استفاده می‌کنیم
FROM python:3.10-alpine

# نصب پکیج‌های ضروری زمان اجرا (که قبلا کامپایل شده‌اند)
# این کار باعث می‌شود ابزارهای build در ایمیج نهایی وجود نداشته باشند
RUN apk add --no-cache libgomp

WORKDIR /app

# ایجاد دایرکتوری برای دیسک پایدار
RUN mkdir -p /app/product_db

# فقط پکیج‌های نصب شده از مرحله قبل را کپی می‌کنیم
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# کد برنامه را کپی می‌کنیم
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
