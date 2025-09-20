import logging
import os
import chromadb
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import gdown

# --- 1. پیکربندی و ثابت‌ها ---
BASE_DATA_DIR = Path("/app/product_db")
DB_PATH = str(BASE_DATA_DIR)
COLLECTION_NAME = "products"
API_VERSION = "2.0.0-headless"
TOP_K_RESULTS = 15

# --- 2. راه‌اندازی لاگ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. متغیرهای سراسری ---
collection = None
app_initialized = False

# --- 4. مدل‌های Pydantic ---
class VectorSearchRequest(BaseModel):
    embedding: List[float] = Field(..., description="The pre-computed embedding vector for the query.")
    keywords: List[str] = Field(..., description="A list of keywords for initial filtering.", example=["rug", "velvet"])

class SearchResult(BaseModel):
    id: str
    name: str
    score: float

class DownloadRequest(BaseModel):
    drive_link: str
    destination_path: str = ""

# --- 5. ساخت اپلیکیشن FastAPI ---
app = FastAPI(
    title="Headless Vector Search API",
    description="An API that receives pre-computed embeddings and performs a hybrid search in ChromaDB.",
    version=API_VERSION
)

# --- 6. توابع همزمان (Sync) ---
def initialize_database_sync():
    """فقط دیتابیس را بارگذاری می‌کند."""
    global collection, app_initialized
    if not os.path.isdir(DB_PATH):
        raise FileNotFoundError("Database path does not exist. Please download the database first.")
    
    logger.info(f"--- در حال اتصال به دیتابیس ChromaDB در مسیر: {DB_PATH} ---")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"✅ اتصال به ChromaDB موفقیت‌آمیز بود. کالکشن '{COLLECTION_NAME}' شامل {collection.count()} آیتم است.")
    app_initialized = True

def search_sync(embedding: List[float], keywords: List[str]) -> List[SearchResult]:
    """جستجوی ترکیبی با استفاده از وکتور آماده."""
    # مرحله ۱: فیلتر اولیه با کلیدواژه
    where_filter = {"$or": [{"$contains": kw} for kw in keywords]} if len(keywords) > 1 else {"$contains": keywords[0]}
    
    # در این مرحله فقط اسناد (documents) و شناسه‌ها (ids) را می‌گیریم، چون به امبدینگ‌هایشان برای مقایسه نیازی نداریم
    results_keyword = collection.get(where_document=where_filter, include=["documents"])
    
    if not results_keyword or not results_keyword.get('ids'):
        return []
        
    # مرحله ۲: جستجوی معنایی روی نتایج فیلتر شده
    # ChromaDB به صورت داخلی شباهت کسینوسی را محاسبه می‌کند
    vector_search_results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K_RESULTS,
        where={"id": {"$in": results_keyword['ids']}} # جستجو فقط روی اسنادی که از فیلتر رد شده‌اند
    )

    if not vector_search_results or not vector_search_results.get('ids')[0]:
        return []

    # آماده‌سازی خروجی نهایی
    final_results = []
    ids = vector_search_results['ids'][0]
    distances = vector_search_results['distances'][0]
    documents = vector_search_results['documents'][0]

    for i in range(len(ids)):
        # تبدیل فاصله (distance) به امتیاز شباهت (similarity score)
        score = (1 - distances[i]) * 100
        final_results.append({
            "id": ids[i],
            "name": documents[i],
            "score": score
        })
        
    return final_results

def start_download(url: str, output_path: Path):
    """تابع دانلود که در پس‌زمینه اجرا می‌شود."""
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
    """Endpoint برای راه‌اندازی دیتابیس."""
    global app_initialized
    if app_initialized:
        return {"status": "warning", "message": "Application is already initialized."}
    try:
        await asyncio.get_running_loop().run_in_executor(None, initialize_database_sync)
        return {"status": "success", "message": "Database initialized successfully."}
    except Exception as e:
        app_initialized = False
        logger.critical(f"❌ راه‌اندازی دیتابیس ناموفق بود: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize database: {str(e)}")

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

@app.post("/vector-search/", response_model=List[SearchResult])
async def vector_search(request: VectorSearchRequest):
    """Endpoint اصلی جستجو که وکتور آماده دریافت می‌کند."""
    if not app_initialized:
        raise HTTPException(status_code=503, detail="Service is not initialized. Please call /startup first.")
    if not request.keywords:
        raise HTTPException(status_code=400, detail="Keywords list cannot be empty.")
    try:
        loop = asyncio.get_running_loop()
        final_results = await loop.run_in_executor(None, search_sync, request.embedding, request.keywords)
        logger.info(f"Returning {len(final_results)} search results.")
        return final_results
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the search process.")

@app.get("/", summary="Health Check")
async def read_root():
    """Endpoint ساده برای بررسی سلامت سرویس."""
    return {
        "status": "OK" if app_initialized else "Pending Initialization",
        "message": "Server is running. Call /startup to initialize model and DB.",
        "version": API_VERSION,
        "initialized": app_initialized
    }
