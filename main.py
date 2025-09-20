import os
import gdown
import asyncio
import logging
import chromadb
import numpy as np
from typing import List
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks


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
KEYWORD_BATCH_SIZE = 100
ID_RERANK_BATCH_SIZE = 500

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
    جستجوی ترکیبی کاملاً دسته‌بندی شده برای جلوگیری از خطای "too many SQL variables"
    بدون از دست دادن هیچ نتیجه‌ای.
    """
    logger.info(f"Performing FULLY BATCHED HYBRID search with {len(keywords)} keywords.")
    
    # --- مرحله ۱: فیلتر کردن با کلیدواژه‌ها به صورت دسته‌بندی شده ---
    all_ids_from_keyword_filter = set()
    keyword_batches = [keywords[i:i + KEYWORD_BATCH_SIZE] for i in range(0, len(keywords), KEYWORD_BATCH_SIZE)]
    
    for batch in keyword_batches:
        where_filter = {"$or": [{"$contains": kw} for kw in batch]} if len(batch) > 1 else {"$contains": batch[0]}
        try:
            # فقط آی‌دی‌ها را می‌گیریم تا حجم داده کمتر باشد
            batch_results = collection.get(where_document=where_filter, include=[])
            if batch_results and batch_results.get('ids'):
                all_ids_from_keyword_filter.update(batch_results['ids'])
        except Exception as e:
            logger.warning(f"A keyword batch query failed, but continuing. Error: {e}")

    if not all_ids_from_keyword_filter:
        logger.warning("No results found after keyword filtering stage.")
        return []
    
    unique_ids = list(all_ids_from_keyword_filter)
    logger.info(f"Found {len(unique_ids)} unique results after keyword filter. Reranking in batches...")

    # --- مرحله ۲: رتبه‌بندی مجدد (جستجوی وکتوری) به صورت دسته‌بندی شده ---
    all_final_results = []
    id_batches = [unique_ids[i:i + ID_RERANK_BATCH_SIZE] for i in range(0, len(unique_ids), ID_RERANK_BATCH_SIZE)]

    for id_batch in id_batches:
        try:
            vector_search_results = collection.query(
                query_embeddings=[embedding],
                n_results=len(id_batch), # تمام نتایج این دسته را می‌خواهیم
                where={"id": {"$in": id_batch}}
            )

            if vector_search_results and vector_search_results.get('ids')[0]:
                ids = vector_search_results['ids'][0]
                distances = vector_search_results['distances'][0]
                documents = vector_search_results['documents'][0]
                for i in range(len(ids)):
                    score = (1 - distances[i]) * 100
                    all_final_results.append({"id": ids[i], "name": documents[i], "score": score})
        except Exception as e:
            logger.warning(f"An ID batch query for reranking failed, but continuing. Error: {e}")
            
    # --- مرحله ۳: مرتب‌سازی نهایی و انتخاب بهترین نتایج ---
    if not all_final_results:
        logger.warning("No results found after reranking stage.")
        return []

    # مرتب‌سازی تمام نتایج جمع‌آوری شده بر اساس امتیاز (score) به صورت نزولی
    all_final_results.sort(key=lambda x: x['score'], reverse=True)
    
    # برگرداندن N نتیجه برتر (TOP_K_RESULTS)
    top_results = all_final_results[:TOP_K_RESULTS]
    
    logger.info(f"Returning {len(top_results)} final hybrid search results after sorting all candidates.")
    return top_results

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


@app.post("/hybrid-search/", response_model=List[SearchResult])
async def hybrid_search(request: VectorSearchRequest):
    logger.info(f"درخواست جستجوی هیبریدی با {len(request.keywords)} کلمه کلیدی دریافت شد.")

    if collection is None:
        raise HTTPException(status_code=503, detail="سرویس در دسترس نیست: دیتابیس بارگذاری نشده است.")

    if not request.keywords:
        raise HTTPException(status_code=400, detail="لیست کلمات کلیدی نمی‌تواند خالی باشد.")

    try:
        # مرحله ۱: فیلتر کردن با کلیدواژه‌ها (بدون تغییر)
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
                pass

        if not all_ids_from_keyword_filter:
            return []
        
        unique_ids = list(all_ids_from_keyword_filter)
        logger.info(f"{len(unique_ids)} نتیجه پس از فیلتر یافت شد. در حال واکشی اطلاعات...")

        # مرحله ۲: گرفتن اطلاعات کامل محصولات فیلتر شده
        results_keyword = collection.get(ids=unique_ids, include=["documents", "embeddings"])

        if not results_keyword or not results_keyword.get('ids'):
            return []

        logger.info(f"اطلاعات کامل برای {len(results_keyword['ids'])} محصول دریافت شد. در حال رتبه‌بندی مجدد...")

        # --- ✨✨✨ جایگزینی محاسبه شباهت با NumPy ✨✨✨ ---
        
        # مرحله ۳: محاسبه شباهت و رتبه‌بندی مجدد با NumPy
        query_embedding = np.array(request.embedding, dtype=np.float32)
        filtered_embeddings = np.array(results_keyword['embeddings'], dtype=np.float32)
        
        # محاسبه شباهت کسینوسی به صورت بهینه
        # 1. نرمالایز کردن وکتورها
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        filtered_norms = filtered_embeddings / np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
        # 2. محاسبه ضرب داخلی (که حالا معادل شباهت کسینوسی است)
        similarities = np.dot(filtered_norms, query_norm)

        reranked_results = []
        for i, doc_name in enumerate(results_keyword['documents']):
            reranked_results.append({
                "id": results_keyword['ids'][i],
                "name": doc_name,
                "score": similarities[i] * 100
            })
        
        # مرحله ۴: مرتب‌سازی و انتخاب بهترین‌ها
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = reranked_results[:TOP_K_RESULTS]
        
        logger.info(f"بازگرداندن {len(top_results)} نتیجه نهایی.")
        return top_results
    
    except Exception as e:
        logger.error(f"خطای غیرمنتظره: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="یک خطای داخلی در سرور رخ داد.")



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
