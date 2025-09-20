import logging
import os
import chromadb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# --- 1. پیکربندی و ثابت‌ها ---
DB_PATH = "/app/product_db"
COLLECTION_NAME = "products"
API_VERSION = "2.1.0-headless-lightweight"
TOP_K_RESULTS = 15
KEYWORD_BATCH_SIZE = 100
ID_RERANK_BATCH_SIZE = 500 # این متغیر در این نسخه استفاده نمی‌شود ولی برای آینده نگه داشته شده

# --- 2. راه‌اندازی لاگ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. بارگذاری دیتابیس ---
try:
    logger.info(f"--- در حال اتصال به ChromaDB در مسیر محلی: {DB_PATH} ---")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    logger.info(f"✅ اتصال به ChromaDB موفقیت‌آمیز بود. کالکشن '{COLLECTION_NAME}' شامل {collection.count()} آیتم است.")

except Exception as e:
    logger.critical(f"❌ خطای بحرانی هنگام اتصال به دیتابیس: {e}", exc_info=True)
    collection = None

# --- 4. مدل‌های Pydantic برای ورودی و خروجی API ---
class VectorSearchRequest(BaseModel):
    embedding: List[float] = Field(..., description="وکتور امبدینگ از پیش محاسبه شده برای کوئری.")
    keywords: List[str] = Field(..., description="لیستی از کلمات کلیدی برای فیلتر اولیه.", example=["فرش", "مخمل"])

class SearchResult(BaseModel):
    id: str = Field(..., description="شناسه منحصر به فرد محصول.")
    name: str = Field(..., description="نام فارسی محصول.")
    score: float = Field(..., description="امتیاز شباهت معنایی به صورت درصد (0-100).")

# --- 5. اپلیکیشن FastAPI ---
app = FastAPI(
    title="Lightweight Headless Hybrid Search API",
    description="یک API سبک که جستجوی ترکیبی را بدون وابستگی به مدل‌های سنگین انجام می‌دهد.",
    version=API_VERSION
)

@app.post("/hybrid-search/", response_model=List[SearchResult])
async def hybrid_search(request: VectorSearchRequest):
    logger.info(f"درخواست جستجوی هیبریدی با {len(request.keywords)} کلمه کلیدی دریافت شد.")

    if collection is None:
        raise HTTPException(status_code=503, detail="سرویس در دسترس نیست: دیتابیس بارگذاری نشده است.")

    if not request.keywords:
        raise HTTPException(status_code=400, detail="لیست کلمات کلیدی نمی‌تواند خالی باشد.")

    try:
        # مرحله ۱: فیلتر کردن با کلیدواژه‌ها به صورت دسته‌بندی شده
        all_ids_from_keyword_filter = set()
        normalized_keywords = [kw.lower() for kw in request.keywords]
        keyword_batches = [normalized_keywords[i:i + KEYWORD_BATCH_SIZE] for i in range(0, len(normalized_keywords), KEYWORD_BATCH_SIZE)]
        
        for batch in keyword_batches:
            where_filter = {"$or": [{"$contains": kw} for kw in batch]} if len(batch) > 1 else {"$contains": batch[0]}
            try:
                batch_results = collection.get(where_document=where_filter, include=[])
                if batch_results and batch_results.get('ids'):
                    all_ids_from_keyword_filter.update(batch_results['ids'])
            except Exception:
                # در صورت بروز خطا در یک دسته، به کار ادامه می‌دهیم
                pass

        if not all_ids_from_keyword_filter:
            logger.warning("هیچ نتیجه‌ای پس از مرحله فیلتر با کلمات کلیدی یافت نشد.")
            return []
        
        unique_ids = list(all_ids_from_keyword_filter)
        logger.info(f"{len(unique_ids)} نتیجه منحصر به فرد پس از فیلتر کلمات کلیدی یافت شد. در حال واکشی اطلاعات کامل...")

        # مرحله ۲: گرفتن اطلاعات کامل (شامل امبدینگ‌ها) برای محصولات فیلتر شده
        results_keyword = collection.get(ids=unique_ids, include=["documents", "embeddings"])

        if not results_keyword or not results_keyword.get('ids'):
            logger.warning("پس از واکشی، هیچ محصولی با آی‌دی‌های مشخص شده یافت نشد.")
            return []

        logger.info(f"اطلاعات کامل برای {len(results_keyword['ids'])} محصول دریافت شد. در حال رتبه‌بندی مجدد در حافظه...")

        # مرحله ۳: محاسبه شباهت و رتبه‌بندی مجدد با NumPy
        query_embedding = np.array(request.embedding, dtype=np.float32)
        filtered_embeddings = np.array(results_keyword['embeddings'], dtype=np.float32)
        
        # محاسبه بهینه شباهت کسینوسی
        # 1. نرمالایز کردن وکتورها (تقسیم بر طولشان)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        filtered_norms = filtered_embeddings / np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
        # 2. محاسبه ضرب داخلی (Dot Product) که حالا معادل شباهت کسینوسی است
        similarities = np.dot(filtered_norms, query_norm)

        reranked_results = []
        for i, doc_name in enumerate(results_keyword['documents']):
            reranked_results.append({
                "id": results_keyword['ids'][i],
                "name": doc_name,
                "score": similarities[i] * 100  # تبدیل شباهت به درصد
            })
        
        # مرحله ۴: مرتب‌سازی نهایی و انتخاب بهترین نتایج
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = reranked_results[:TOP_K_RESULTS]
        
        logger.info(f"بازگرداندن {len(top_results)} نتیجه نهایی پس از رتبه‌بندی مجدد.")
        return top_results
    
    except Exception as e:
        logger.error(f"خطای غیرمنتظره در حین جستجوی هیبریدی: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="یک خطای داخلی در سرور رخ داد.")

@app.get("/", summary="Health Check")
async def read_root():
    return {
        "status": "OK" if collection else "Error",
        "message": "Headless hybrid search server is running." if collection else "Database not loaded.",
        "version": API_VERSION
    }
