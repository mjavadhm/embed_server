import logging
import os
import torch
import numpy as np
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer, util

# --- 1. Configuration & Constants ---
DB_PATH = "/app/product_db"
MODEL_PATH = "/app/product_db/model"
COLLECTION_NAME = "products"
API_VERSION = "0.1.0"
TOP_K_RESULTS = 15

# --- 2. Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Model & Database Loading ---
try:
    logger.info("--- Initializing application components from local paths ---")
    device = "cpu"
    logger.info(f"Loading sentence transformer model from local path: {MODEL_PATH} onto device: '{device}'")
    model = SentenceTransformer(MODEL_PATH, device=device)
    logger.info("✅ Embedding model loaded successfully.")
    logger.info(f"Connecting to ChromaDB at local path: {DB_PATH}")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    logger.info(f"✅ Successfully connected to ChromaDB. Collection '{COLLECTION_NAME}' contains {collection.count()} items.")
except Exception as e:
    logger.critical(f"❌ Critical error during component initialization: {e}", exc_info=True)
    collection = None
    model = None

# --- 4. Pydantic Models for API I/O ---
class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="The text query for semantic search.", example="فرش آشپزخانه مخملی")

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="The full-text query for semantic search.", example="velvet kitchen rug")
    keywords: List[str] = Field(..., description="A list of keywords for initial filtering.", example=["rug", "velvet"])

class SearchResult(BaseModel):
    id: str = Field(..., description="Unique identifier of the product.")
    name: str = Field(..., description="The Persian name of the product.")
    score: float = Field(..., description="The semantic similarity score for ranking.")

# --- 5. FastAPI Application ---
app = FastAPI(
    title="Hybrid and Semantic Product Search API (Self-Contained)",
    description="An API that performs a two-stage hybrid search and a pure semantic search from a self-contained Docker appliance.",
    version=API_VERSION
)

# =================================================================================
# تابع بهینه‌شده برای جستجوی سریع
# =================================================================================
@app.post("/semantic-search/", response_model=List[SearchResult])
def semantic_search(request: SemanticSearchRequest):
    """
    Performs an efficient pure semantic search using the database's native query capabilities.
    """
    logger.info(f"Received semantic search request. Query: '{request.query}'")

    if collection is None or model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. وکتور کوئری را ایجاد کن
        query_embedding = model.encode(request.query).tolist()

        # 2. از collection.query برای جستجوی بهینه و سریع استفاده کن
        # این تابع از ایندکس‌های ChromaDB برای یافتن سریع نتایج استفاده می‌کند
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RESULTS,
            include=["documents", "distances"]
        )

        # 3. نتایج را برای خروجی فرمت‌بندی کن
        final_results = []
        if results and results.get('ids'):
            ids = results['ids'][0]
            documents = results['documents'][0]
            distances = results['distances'][0]

            for i in range(len(ids)):
                # نکته: فاصله (distance) بستگی به متریک دیتابیس دارد (پیش‌فرض L2)
                # برای تبدیل فاصله به امتیاز (که هرچه بیشتر بهتر باشد)، از یک فرمول ساده استفاده می‌کنیم
                score = 100 / (1 + distances[i])

                final_results.append({
                    "id": ids[i],
                    "name": documents[i],
                    "score": score
                })
        
        logger.info(f"Returning {len(final_results)} semantic search results from DB query.")
        return final_results

    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during semantic search.")

# ... (بقیه کدها شامل hybrid_search و read_root بدون تغییر باقی می‌مانند) ...
@app.post("/hybrid-search/", response_model=List[SearchResult])
def hybrid_search(request: HybridSearchRequest):
    logger.info(f"Received search request. Query: '{request.query}', Keywords: {request.keywords}")

    if collection is None or model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")

    if not request.keywords:
        raise HTTPException(status_code=400, detail="Keywords list cannot be empty.")

    try:
        where_filter = {}
        if len(request.keywords) == 1:
            where_filter = {"$contains": request.keywords[0]}
        else:
            where_filter = {"$or": [{"$contains": kw} for kw in request.keywords]}
        
        results_keyword = collection.get(
            where_document=where_filter,
            include=["documents", "embeddings"]
        )
    except Exception as e:
        logger.error(f"Error during ChromaDB keyword filtering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during database filtering.")
    
    if not results_keyword or not results_keyword.get('ids'):
        return []

    logger.info(f"Found {len(results_keyword['ids'])} results after keyword filtering. Proceeding to re-ranking.")

    full_query_embedding = model.encode(request.query)
    filtered_embeddings = np.array(results_keyword['embeddings'], dtype=np.float32)
    similarities = util.cos_sim(full_query_embedding, filtered_embeddings)

    reranked_results = []
    for i, doc_name in enumerate(results_keyword['documents']):
        reranked_results.append({
            "id": results_keyword['ids'][i],
            "name": doc_name,
            "score": similarities[0][i].item() * 100
        })
    
    reranked_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = reranked_results[:TOP_K_RESULTS]
    logger.info(f"Returning {len(final_results)} re-ranked results.")
    
    return final_results

@app.get("/", summary="Health Check")
def read_root():
    return {
        "status": "OK",
        "message": "Hybrid search server is running.",
        "version": API_VERSION
    }
