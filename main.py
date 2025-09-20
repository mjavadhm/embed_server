import logging
import os
import torch
import numpy as np
import chromadb
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer, util
import gdown
from pathlib import Path

# --- 1. Configuration & Constants ---
# مسیر پایه که به دیسک متصل است
BASE_DATA_DIR = Path("/app/product_db") 

# مسیرهای مدل و دیتابیس هر دو داخل مسیر پایه هستند
DB_PATH = str(BASE_DATA_DIR)
MODEL_PATH = str(BASE_DATA_DIR / "embedding_model") # ✨ تغییر اصلی اینجاست ✨
COLLECTION_NAME = "products"
API_VERSION = "1.1.0"
TOP_K_RESULTS = 15

# --- 2. Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Global Variables for Model & DB ---
model: SentenceTransformer = None
collection = None
app_initialized = False

# --- 4. Pydantic Models for API I/O ---
class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="The full-text query for semantic search.", example="velvet kitchen rug")
    keywords: List[str] = Field(..., description="A list of keywords for initial filtering.", example=["rug", "velvet"])

class SearchResult(BaseModel):
    id: str = Field(..., description="Unique identifier of the product.")
    name: str = Field(..., description="The Persian name of the product.")
    score: float = Field(..., description="The semantic similarity score as a percentage (0-100).")

class DownloadRequest(BaseModel):
    drive_link: str = Field(..., description="لینک اشتراک‌گذاری فایل در گوگل درایو")
    # مسیر مقصد اکنون اختیاری است و به صورت نسبی به مسیر پایه در نظر گرفته می‌شود
    destination_path: str = Field("", description="مسیر نسبی برای ذخیره فایل روی دیسک (مثال: my_model/ یا my_database/)")

# --- 5. FastAPI Application ---
app = FastAPI(
    title="Unified Hybrid Search and Downloader API",
    description="An API that first downloads files, then initializes a search model, and finally performs hybrid search.",
    version=API_VERSION
)

# --- 6. Helper Functions ---
def initialize_components():
    """Loads the embedding model and connects to the ChromaDB database."""
    global model, collection, app_initialized
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(DB_PATH):
            raise FileNotFoundError("Model or Database path does not exist. Please download files first.")

        logger.info("--- Initializing application components from local paths ---")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading sentence transformer model from: {MODEL_PATH} onto device: '{device}'")
        model = SentenceTransformer(MODEL_PATH, device=device)
        logger.info("✅ Embedding model loaded successfully.")

        logger.info(f"Connecting to ChromaDB at: {DB_PATH}")
        db_client = chromadb.PersistentClient(path=DB_PATH)
        collection = db_client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"✅ Successfully connected to ChromaDB. Collection '{COLLECTION_NAME}' contains {collection.count()} items.")
        
        app_initialized = True

    except Exception as e:
        logger.critical(f"❌ Critical error during component initialization: {e}", exc_info=True)
        model = None
        collection = None
        app_initialized = False
        raise e

def start_download(url: str, output_path: Path):
    """Background task to download a file from Google Drive."""
    logger.info(f"🚀 Starting download from {url} to {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # استفاده از gdown برای دانلود (تشخیص می‌دهد فایل است یا پوشه)
        gdown.download(url=url, output=str(output_path), quiet=False, fuzzy=True)
        logger.info(f"✅ Download complete for: {output_path}")
    except Exception as e:
        logger.error(f"❌ Error downloading file {url}. Reason: {e}")

# --- 7. API Endpoints ---
@app.post("/startup/")
def startup_server():
    """
    Initializes the embedding model and the database connection.
    This must be called after files are downloaded and before searching.
    """
    if app_initialized:
        return {"status": "warning", "message": "Application is already initialized."}
    
    try:
        initialize_components()
        return {"status": "success", "message": "Application components initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize components: {str(e)}")

@app.post("/get-files/")
def schedule_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Schedules a file or folder download from Google Drive to the persistent disk."""
    # مسیر کامل مقصد را با ترکیب مسیر پایه و مسیر درخواستی کاربر می‌سازیم
    full_path = BASE_DATA_DIR.joinpath(request.destination_path).resolve()

    # جلوگیری از حملات Path Traversal
    if BASE_DATA_DIR not in full_path.parents and full_path != BASE_DATA_DIR:
        raise HTTPException(
            status_code=400,
            detail="Error: Invalid destination path. Must be inside the base data directory."
        )

    # وظیفه دانلود را به پس‌زمینه اضافه می‌کنیم
    background_tasks.add_task(start_download, request.drive_link, full_path)

    return {
        "status": "success",
        "message": "Download task has been scheduled.",
        "details": {
            "drive_link": request.drive_link,
            "save_location": str(full_path)
        }
    }

@app.post("/hybrid-search/", response_model=List[SearchResult])
def hybrid_search(request: HybridSearchRequest):
    # (بدون تغییر نسبت به نسخه قبل)
    ...

@app.get("/", summary="Health Check")
def read_root():
    # (بدون تغییر نسبت به نسخه قبل)
    ...
