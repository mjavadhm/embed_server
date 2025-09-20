import logging
import os
import chromadb
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import gdown

# --- 1. پیکربندی و ثابت‌ها ---
BASE_DATA_DIR = Path("/app/product_db")
DB_PATH = str(BASE_DATA_DIR)
COLLECTION_NAME = "products"
API_VERSION = "2.3.0-final-debug"
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

class DebugRequest(BaseModel):
    product_name: str = Field(..., description="نام دقیق فارسی محصول برای جستجو.")

class DebugResponse(BaseModel):
    product_name: str
    embedding: List[float]


# --- 5. ساخت اپلیکیشن FastAPI ---
app = FastAPI(
    title="Headless Vector Search API (Debug & Production Ready)",
    description="Provides endpoints for pure vector search, hybrid search, and debugging.",
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


def search_sync_pure_vector(embedding: List[float]) -> List[SearchResult]:
    """
    جستجوی وکتوری خالص را انجام می‌دهد و فیلتر کلیدواژه را نادیده می‌گیرد.
    """
    logger.info("Performing PURE vector search (keyword filter disabled).")
    vector_search_results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K_RESULTS,
    )

    if not vector_search_results or not vector_search_results.get('ids')[0]:
        logger.warning("No results found from pure vector search.")
        return []

    final_results = []
    ids, distances, documents = vector_search_results['ids'][0], vector_search_results['distances'][0], vector_search_results['documents'][0]
    for i in range(len(ids)):
        score = (1 - distances[i]) * 100
        final_results.append({"id": ids[i], "name": documents[i] if documents else "N/A", "score": score})
    return final_results


def search_sync_hybrid(embedding: List[float], keywords: List[str]) -> List[SearchResult]:
    """
    جستجوی ترکیبی با تقسیم کلیدواژه‌ها به دسته‌های کوچک‌تر برای جلوگیری از خطای دیتابیس.
    """
    logger.info(f"Performing HYBRID search with {len(keywords)} keywords.")
    
    # اگر تعداد کلیدواژه‌ها کم است، از روش قبلی استفاده کن
    if len(keywords) <= KEYWORD_BATCH_SIZE:
        where_filter = {"$or": [{"$contains": kw} for kw in keywords]} if len(keywords) > 1 else {"$contains": keywords[0]}
        results_keyword = collection.get(where_document=where_filter, include=["documents"])
    else:
        # اگر تعداد کلیدواژه‌ها زیاد است، آن‌ها را دسته‌بندی کن
        logger.info(f"Keyword count exceeds batch size. Splitting into chunks of {KEYWORD_BATCH_SIZE}.")
        all_ids = set()
        
        # تقسیم لیست کلیدواژه‌ها به دسته‌های کوچک‌تر
        keyword_batches = [keywords[i:i + KEYWORD_BATCH_SIZE] for i in range(0, len(keywords), KEYWORD_BATCH_SIZE)]
        
        for batch in keyword_batches:
            logger.info(f"Processing keyword batch with {len(batch)} items.")
            where_filter = {"$or": [{"$contains": kw} for kw in batch]}
            try:
                batch_results = collection.get(where_document=where_filter, include=[]) # فقط آی‌دی‌ها را لازم داریم
                if batch_results and batch_results.get('ids'):
                    all_ids.update(batch_results['ids'])
            except Exception as e:
                logger.warning(f"A batch query failed, but continuing. Error: {e}")

        if not all_ids:
            logger.warning("No results found after keyword filtering stage.")
            return []
            
        # ساخت یک دیکشنری نتیجه ساختگی برای سازگاری با بقیه کد
        results_keyword = {'ids': list(all_ids)}

    if not results_keyword or not results_keyword.get('ids'):
        logger.warning("No results found after keyword filtering stage.")
        return []
        
    logger.info(f"Found {len(results_keyword['ids'])} unique results after keyword filter. Reranking...")
    
    # مرحله نهایی: جستجوی وکتوری روی آی‌دی‌های پیدا شده
    # نکته: اگر تعداد آی‌دی‌ها خیلی زیاد باشد (مثلا چند ده هزار)، این بخش هم ممکن است کند شود
    # اما معمولا بسیار سریع‌تر از جستجوی متنی است.
    vector_search_results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K_RESULTS,
        where={"id": {"$in": results_keyword['ids']}}
    )

    if not vector_search_results or not vector_search_results.get('ids')[0]:
        logger.warning("No results found after reranking stage.")
        return []

    final_results = []
    ids, distances, documents = vector_search_results['ids'][0], vector_search_results['distances'][0], vector_search_results['documents'][0]
    for i in range(len(ids)):
        score = (1 - distances[i]) * 100
        final_results.append({"id": ids[i], "name": documents[i], "score": score})
    return final_results

def get_embedding_by_name_sync(product_name: str) -> Optional[List[float]]:
    """
    وکتور امبدینگ یک محصول را با استفاده از نام دقیق آن از دیتابیس استخراج می‌کند.
    """
    logger.info(f"Debugging: Attempting to retrieve product by name: '{product_name}'")
    result = collection.get(
        where_document={"$eq": product_name},
        include=["embeddings"]
    )
    if result and result.get('embeddings'):
        logger.info(f"✅ Debug: Product '{product_name}' found.")
        return result['embeddings'][0]
    else:
        logger.warning(f"⚠️ Debug: Product '{product_name}' NOT found.")
        return None



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
    """Endpoint برای زمان‌بندی دانلود."""
    full_path = BASE_DATA_DIR.joinpath(request.destination_path).resolve()
    if BASE_DATA_DIR not in full_path.parents and full_path != BASE_DATA_DIR:
        raise HTTPException(status_code=400, detail="خطا: مسیر مقصد نامعتبر است.")
    background_tasks.add_task(start_download, request.drive_link, full_path)
    return {
        "status": "success",
        "message": "وظیفه دانلود با موفقیت زمان‌بندی شد.",
        "details": {"drive_link": request.drive_link, "save_location": str(full_path)}
    }


@app.post("/vector-search/", response_model=List[SearchResult], summary="Pure Vector Search (Debug)")
async def vector_search(request: VectorSearchRequest):
    """
    Endpoint برای جستجوی وکتوری خالص (فیلتر کلیدواژه را نادیده می‌گیرد).
    """
    if not app_initialized:
        raise HTTPException(status_code=503, detail="Service is not initialized. Please call /startup first.")
    
    try:
        loop = asyncio.get_running_loop()
        final_results = await loop.run_in_executor(None, search_sync_pure_vector, request.embedding)
        logger.info(f"Returning {len(final_results)} pure vector search results.")
        return final_results
    except Exception as e:
        logger.error(f"Error during pure vector search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the search process.")


@app.post("/hybrid-search/", response_model=List[SearchResult], summary="Hybrid Search (Production)")
async def hybrid_search(request: VectorSearchRequest):
    """Endpoint اصلی جستجوی ترکیبی (ابتدا کلیدواژه، سپس وکتور)."""
    if not app_initialized:
        raise HTTPException(status_code=503, detail="Service is not initialized. Please call /startup first.")
    if not request.keywords:
        raise HTTPException(status_code=400, detail="Keywords list cannot be empty.")
    try:
        # ✨✨✨ اصلاح اصلی اینجاست: کلمات کلیدی را به حروف کوچک تبدیل کنید ✨✨✨
        normalized_keywords = [kw.lower() for kw in request.keywords]

        loop = asyncio.get_running_loop()
        # از لیست نرمال‌شده در تابع جستجو استفاده کنید
        final_results = await loop.run_in_executor(None, search_sync_hybrid, request.embedding, normalized_keywords)
        
        logger.info(f"Returning {len(final_results)} hybrid search results.")
        return final_results
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the search process.")


@app.post("/get-embedding/", response_model=DebugResponse, summary="Debug: Get Embedding by Name")
async def get_embedding_by_name(request: DebugRequest):
    """
    یک endpoint برای تست که وکتور ذخیره شده برای یک محصول را با نام دقیق آن برمی‌گرداند.
    """
    if not app_initialized:
        raise HTTPException(status_code=503, detail="Service is not initialized. Please call /startup first.")
    try:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, get_embedding_by_name_sync, request.product_name)
        if embedding:
            return {"product_name": request.product_name, "embedding": embedding}
        else:
            raise HTTPException(status_code=404, detail=f"Product '{request.product_name}' not found in the database.")
    except Exception as e:
        logger.error(f"Error during debug lookup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the debug lookup.")



@app.get("/", summary="Health Check")
async def read_root():
    """Endpoint ساده برای بررسی سلامت سرویس."""
    return {
        "status": "OK" if app_initialized else "Pending Initialization",
        "message": "Server is running. Call /startup to initialize model and DB.",
        "version": API_VERSION,
        "initialized": app_initialized
    }

