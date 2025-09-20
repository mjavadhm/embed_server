import logging
import os
import torch
import numpy as np
import chromadb
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer, util
import gdown
from pathlib import Path

# --- 1. پیکربندی و ثابت‌ها ---
BASE_DATA_DIR = Path("/app/product_db")
DB_PATH = str(BASE_DATA_DIR)
MODEL_PATH = str(BASE_DATA_DIR / "embedding_model")
COLLECTION_NAME = "products"
API_VERSION = "1.2.0-async"
TOP_K_RESULTS = 15

# --- 2. راه‌اندازی لاگ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. متغیرهای سراسری برای مدل و دیتابیس ---
model: SentenceTransformer = None
collection = None
app_initialized = False

# --- 4. مدل‌های Pydantic ---
class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="The full-text query for semantic search.", example="velvet kitchen rug")
    keywords: List[str] = Field(..., description="A list of keywords for initial filtering.", example=["rug", "velvet"])

class SearchResult(BaseModel):
    id: str = Field(..., description="Unique identifier of the product.")
    name: str = Field(..., description="The Persian name of the product.")
    score: float = Field(..., description="The semantic similarity score as a percentage (0-100).")

class DownloadRequest(BaseModel):
    drive_link: str = Field(..., description="لینک اشتراک‌گذاری فایل در گوگل درایو")
    destination_path: str = Field("", description="مسیر نسبی برای ذخیره فایل روی دیسک (مثال: embedding_model)")

# --- 5. ساخت اپلیکیشن FastAPI ---
app = FastAPI(
    title="Async Unified Hybrid Search and Downloader API",
    description="An asynchronous API that downloads files, initializes a search model, and performs hybrid search.",
    version=API_VERSION
)

# --- 6. توابع همزمان (Sync) برای اجرا در Thread Pool ---
def initialize_components_sync():
    """
    نسخه همزمان (sync) تابع راه‌اندازی که شامل کدهای blocking است.
    این تابع در یک thread جداگانه اجرا خواهد شد.
    """
    global model, collection, app_initialized
    if not os.path.exists(MODEL_PATH) or not os.path.isdir(DB_PATH):
        raise FileNotFoundError("Model or Database path does not exist. Please download files first.")
    
    logger.info("--- در حال راه‌اندازی کامپوننت‌های برنامه از مسیرهای محلی ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"در حال بارگذاری مدل از مسیر: {MODEL_PATH} روی دستگاه: '{device}'")
    model = SentenceTransformer(MODEL_PATH, device=device)
    logger.info("✅ مدل امبدینگ با موفقیت بارگذاری شد.")

    logger.info(f"در حال اتصال به دیتابیس ChromaDB در مسیر: {DB_PATH}")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"✅ اتصال به ChromaDB موفقیت‌آمیز بود. کالکشن '{COLLECTION_NAME}' شامل {collection.count()} آیتم است.")
    
    app_initialized = True

def search_sync(query: str, keywords: List[str]) -> List[SearchResult]:
    """
    نسخه همزمان (sync) تابع جستجو. تمام کدهای سنگین CPU-bound اینجا هستند.
    این تابع در یک thread جداگانه اجرا خواهد شد.
    """
    # فیلتر بر اساس کلیدواژه
    where_filter = {"$or": [{"$contains": kw} for kw in keywords]} if len(keywords) > 1 else {"$contains": keywords[0]}
    results_keyword = collection.get(where_document=where_filter, include=["documents", "embeddings"])
    
    if not results_keyword or not results_keyword.get('ids'):
        logger.info("پس از فیلتر کلیدواژه نتیجه‌ای یافت نشد.")
        return []

    logger.info(f"تعداد {len(results_keyword['ids'])} نتیجه پس از فیلتر کلیدواژه یافت شد. در حال رتبه‌بندی مجدد...")

    # رتبه‌بندی مجدد با جستجوی معنایی (بخش سنگین)
    full_query_embedding = model.encode(query)
    filtered_embeddings = np.array(results_keyword['embeddings'], dtype=np.float32)
    similarities = util.cos_sim(full_query_embedding, filtered_embeddings)

    reranked_results = [
        {
            "id": results_keyword['ids'][i],
            "name": doc_name,
            "score": similarities[0][i].item() * 100
        }
        for i, doc_name in enumerate(results_keyword['documents'])
    ]
    
    reranked_results.sort(key=lambda x: x['score'], reverse=True)
    return reranked_results[:TOP_K_RESULTS]

def start_download(url: str, output_path: Path):
    """تابع دانلود که در پس‌زمینه اجرا می‌شود (این بخش نیازی به async ندارد)."""
    # ... (بدون تغییر)
    logger.info(f"🚀 شروع دانلود از {url} به {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url=url, output=str(output_path), quiet=False, fuzzy=True)
        logger.info(f"✅ دانلود برای {output_path} کامل شد.")
    except Exception as e:
        logger.error(f"❌ خطا در دانلود فایل {url}. دلیل: {e}")

# --- 7. Endpoint های API (Asynchronous) ---
@app.post("/startup/")
async def startup_server():
    """
    Endpoint غیرهمزمان (async) برای راه‌اندازی مدل و دیتابیس.
    """
    global app_initialized
    if app_initialized:
        return {"status": "warning", "message": "Application is already initialized."}
    
    try:
        # اجرای تابع blocking در یک thread جداگانه
        await asyncio.get_running_loop().run_in_executor(None, initialize_components_sync)
        return {"status": "success", "message": "کامپوننت‌های برنامه با موفقیت راه‌اندازی شدند."}
    except Exception as e:
        # ریست کردن وضعیت در صورت بروز خطا
        app_initialized = False
        logger.critical(f"❌ راه‌اندازی ناموفق بود: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"راه‌اندازی ناموفق بود: {str(e)}")

@app.post("/get-files/")
async def schedule_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Endpoint برای زمان‌بندی دانلود (این بخش می‌تواند async باشد ولی ضروری نیست)."""
    # ... (بدون تغییر)
    full_path = BASE_DATA_DIR.joinpath(request.destination_path).resolve()
    if BASE_DATA_DIR not in full_path.parents and full_path != BASE_DATA_DIR:
        raise HTTPException(
            status_code=400,
            detail="خطا: مسیر مقصد نامعتبر است."
        )
    background_tasks.add_task(start_download, request.drive_link, full_path)
    return {
        "status": "success",
        "message": "وظیفه دانلود با موفقیت زمان‌بندی شد.",
        "details": {
            "drive_link": request.drive_link,
            "save_location": str(full_path)
        }
    }

@app.post("/hybrid-search/", response_model=List[SearchResult])
async def hybrid_search(request: HybridSearchRequest):
    """
    ✨ Endpoint اصلی جستجو که به صورت غیرهمزمان (async) اجرا می‌شود. ✨
    """
    if not app_initialized or collection is None or model is None:
        raise HTTPException(status_code=503, detail="سرویس در دسترس نیست. لطفاً ابتدا /startup را فراخوانی کنید.")

    if not request.keywords:
        raise HTTPException(status_code=400, detail="لیست کلیدواژه‌ها نمی‌تواند خالی باشد.")

    try:
        # اجرای تابع سنگین و همزمانِ جستجو در یک thread جداگانه
        # این کار از مسدود شدن event loop جلوگیری می‌کند
        loop = asyncio.get_running_loop()
        final_results = await loop.run_in_executor(
            None,  # استفاده از thread pool پیش‌فرض
            search_sync,  # تابع همزمانی که باید اجرا شود
            request.query,  # آرگومان‌های تابع
            request.keywords
        )
        logger.info(f"بازگرداندن {len(final_results)} نتیجه نهایی.")
        return final_results
    except Exception as e:
        logger.error(f"خطا در حین جستجو: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="خطایی در حین فرآیند جستجو رخ داد.")

@app.get("/", summary="Health Check")
async def read_root():
    """Endpoint ساده برای بررسی سلامت سرویس."""
    return {
        "status": "OK" if app_initialized else "Pending Initialization",
        "message": "Server is running. Call /startup to initialize model and DB.",
        "version": API_VERSION,
        "initialized": app_initialized
    }
