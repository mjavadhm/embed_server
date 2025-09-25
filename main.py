import logging
import os
import torch
import numpy as np
import chromadb
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import uuid4
from sentence_transformers import SentenceTransformer, util

# --- 1. Configuration & Constants ---
DB_PATH = "/app/product_db"
MODEL_PATH = "/app/product_db/model"
PRODUCT_COLLECTION_NAME = "products" # Renamed for clarity
CATEGORY_COLLECTION_NAME = "categories" # New constant for categories
API_VERSION = "0.1.1" # Incremented version
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

    # Load or create product collection
    product_collection = db_client.get_or_create_collection(name=PRODUCT_COLLECTION_NAME)
    logger.info(f"✅ Successfully connected to ChromaDB. Collection '{PRODUCT_COLLECTION_NAME}' contains {product_collection.count()} items.")

    # Load or create category collection
    category_collection = db_client.get_or_create_collection(name=CATEGORY_COLLECTION_NAME)
    logger.info(f"✅ Successfully connected to ChromaDB. Collection '{CATEGORY_COLLECTION_NAME}' contains {category_collection.count()} items.")

except Exception as e:
    logger.critical(f"❌ Critical error during component initialization: {e}", exc_info=True)
    product_collection = None
    category_collection = None
    model = None

# --- 4. Pydantic Models for API I/O ---

# Models for Product Search
class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="The text query for semantic search.", example="فرش آشپزخانه مخملی")

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="The full-text query for semantic search.", example="velvet kitchen rug")
    keywords: List[str] = Field(..., description="A list of keywords for initial filtering.", example=["rug", "velvet"])

class SearchResult(BaseModel):
    id: str = Field(..., description="Unique identifier of the product.")
    name: str = Field(..., description="The Persian name of the product.")
    score: float = Field(..., description="The semantic similarity score for ranking.")

# Models for Category Management
class CategoryAddRequest(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional unique identifier for the category. If not provided, a new UUID will be generated.", example="cat_123")
    title: str = Field(..., description="The title of the category, which will be embedded for searching.", example="لوازم خانگی برقی")
    feature_schema: Dict[str, Any] = Field(..., description="A JSON schema describing the features for this category.", example={"brand": "string", "power_watts": "integer"})

class CategorySearchRequest(BaseModel):
    query: str = Field(..., description="The text query to search for categories.", example="وسایل آشپزخانه")

class CategorySearchResult(BaseModel):
    id: str = Field(..., description="Unique identifier of the category.")
    title: str = Field(..., description="The title of the category.")
    feature_schema: Dict[str, Any] = Field(..., description="The feature schema of the category.")
    score: float = Field(..., description="The semantic similarity score for ranking.")


# --- 5. FastAPI Application ---
app = FastAPI(
    title="Hybrid and Semantic Product Search API (Self-Contained)",
    description="An API that performs hybrid search, semantic search, and category management from a self-contained Docker appliance.",
    version=API_VERSION
)

# =================================================================================
# اندپوینت‌های جدید برای مدیریت و جستجوی دسته‌بندی‌ها (Categories)
# =================================================================================

@app.post("/add-category/", summary="Add or Update a Category", status_code=201)
def add_category(request: CategoryAddRequest):
    """
    Adds a new category or updates an existing one based on the ID.
    The category's 'title' is embedded and stored for semantic search.
    """
    logger.info(f"Received request to add/update category. Title: '{request.title}'")

    if category_collection is None or model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")

    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Category title cannot be empty.")

    try:
        # 1. شناسه را مدیریت کن: اگر ارائه نشده بود، یک UUID جدید بساز
        category_id = request.id if request.id else str(uuid4())

        # 2. عنوان را امبد کن
        title_embedding = model.encode(request.title).tolist()

        # 3. داده‌ها را در کالکشن دسته‌بندی‌ها اضافه یا به‌روزرسانی کن (upsert)
        # نکته: متادیتا فقط از انواع داده پایه پشتیبانی می‌کند، بنابراین دیکشنری به رشته JSON تبدیل می‌شود
        category_collection.upsert(
            ids=[category_id],
            embeddings=[title_embedding],
            documents=[request.title],
            metadatas=[{
                "title": request.title,
                "feature_schema": json.dumps(request.feature_schema)
            }]
        )

        logger.info(f"Successfully upserted category with ID: {category_id}")
        return {"status": "success", "id": category_id, "message": "Category added or updated successfully."}

    except Exception as e:
        logger.error(f"Error during adding/updating category: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the category.")


@app.post("/search-category/", response_model=List[CategorySearchResult], summary="Search for Categories")
def search_category(request: CategorySearchRequest):
    """
    Performs a semantic search to find the most relevant categories based on a query.
    """
    logger.info(f"Received category search request. Query: '{request.query}'")

    if category_collection is None or model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. کوئری را امبد کن
        query_embedding = model.encode(request.query).tolist()

        # 2. با استفاده از کوئری امبد شده، در کالکشن دسته‌بندی‌ها جستجو کن
        results = category_collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RESULTS,
            include=["metadatas", "distances"]
        )

        # 3. نتایج را فرمت‌بندی کن
        final_results = []
        if results and results.get('ids'):
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            for i in range(len(ids)):
                score = 100 / (1 + distances[i])
                
                # رشته JSON ذخیره شده در متادیتا را به دیکشنری تبدیل کن
                try:
                    feature_schema_dict = json.loads(metadatas[i].get("feature_schema", "{}"))
                except (json.JSONDecodeError, TypeError):
                    feature_schema_dict = {}

                final_results.append({
                    "id": ids[i],
                    "title": metadatas[i].get("title", ""),
                    "feature_schema": feature_schema_dict,
                    "score": score
                })

        logger.info(f"Returning {len(final_results)} category search results.")
        return final_results

    except Exception as e:
        logger.error(f"Error during category search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during category search.")


# =================================================================================
# اندپوینت‌های موجود برای جستجوی محصولات (با تغییر نام متغیر کالکشن)
# =================================================================================

@app.post("/semantic-search/", response_model=List[SearchResult])
def semantic_search(request: SemanticSearchRequest):
    """
    Performs an efficient pure semantic search using the database's native query capabilities.
    """
    logger.info(f"Received semantic search request. Query: '{request.query}'")

    if product_collection is None or model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        query_embedding = model.encode(request.query).tolist()
        results = product_collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RESULTS,
            include=["documents", "distances"]
        )
        final_results = []
        if results and results.get('ids'):
            ids = results['ids'][0]
            documents = results['documents'][0]
            distances = results['distances'][0]
            for i in range(len(ids)):
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


@app.post("/hybrid-search/", response_model=List[SearchResult])
def hybrid_search(request: HybridSearchRequest):
    logger.info(f"Received search request. Query: '{request.query}', Keywords: {request.keywords}")
    if product_collection is None or model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")
    if not request.keywords:
        raise HTTPException(status_code=400, detail="Keywords list cannot be empty.")
    try:
        where_filter = {"$or": [{"$contains": kw} for kw in request.keywords]} if len(request.keywords) > 1 else {"$contains": request.keywords[0]}
        results_keyword = product_collection.get(where_document=where_filter, include=["documents", "embeddings"])
    except Exception as e:
        logger.error(f"Error during ChromaDB keyword filtering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during database filtering.")
    if not results_keyword or not results_keyword.get('ids'):
        return []
    logger.info(f"Found {len(results_keyword['ids'])} results after keyword filtering. Proceeding to re-ranking.")
    full_query_embedding = model.encode(request.query)
    filtered_embeddings = np.array(results_keyword['embeddings'], dtype=np.float32)
    similarities = util.cos_sim(full_query_embedding, filtered_embeddings)
    reranked_results = [{"id": results_keyword['ids'][i], "name": doc_name, "score": similarities[0][i].item() * 100} for i, doc_name in enumerate(results_keyword['documents'])]
    reranked_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = reranked_results[:TOP_K_RESULTS]
    logger.info(f"Returning {len(final_results)} re-ranked results.")
    return final_results

@app.get("/", summary="Health Check")
def read_root():
    return {
        "status": "OK",
        "message": "Hybrid search server is running.",
        "version": API_VERSION,
        "collections": {
            PRODUCT_COLLECTION_NAME: f"{product_collection.count() if product_collection else 'N/A'} items",
            CATEGORY_COLLECTION_NAME: f"{category_collection.count() if category_collection else 'N/A'} items"
        }
    }
