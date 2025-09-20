# --- مرحله ۱: بیلد (Builder Stage) ---
# از ایمیج Alpine برای نصب استفاده می‌کنیم
FROM python:3.10-alpine as builder

# ✨ تغییر اصلی اینجاست: نصب ابزارهای مورد نیاز برای کامپایل پکیج‌ها ✨
RUN apk add --no-cache build-base

WORKDIR /app

# کپی و نصب نیازمندی‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- مرحله ۲: نهایی (Final Stage) ---
# از ایمیج بسیار سبک Alpine برای اجرای برنامه استفاده می‌کنیم
FROM python:3.10-alpine

# نصب پکیج‌های ضروری زمان اجرا (که قبلا کامپایل شده‌اند)
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
