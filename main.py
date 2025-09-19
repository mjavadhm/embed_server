import logging
import os
import zipfile
import torch
import numpy as np
import chromadb
import gdown
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer, util

# --- 1. Configuration & Constants ---
CHROMA_PATH = "/app/product_db"
COLLECTION_NAME = "products"
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
API_VERSION = "0.1.0"
TOP_K_RESULTS = 15
GDRIVE_FOLDER_ID = "1-tktqeXhjjvpACRWHd9xE2UgDzpt4aco"

# --- 2. Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Automated Database Setup ---
def setup_database():
    """
    Checks for the database. If not present, downloads and extracts it from Google Drive.
    """
    check_file = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    
    if os.path.exists(check_file):
        logger.info("✅ Database found. Skipping download.")
        return

    logger.warning("🟡 Database not found. Initializing download from Google Drive...")
    
    zip_path = os.path.join(CHROMA_PATH, "product_db.zip")
    
    try:
        # Create the target directory if it doesn't exist
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Download the folder from Google Drive
        logger.info(f"Downloading folder with ID: {GDRIVE_FOLDER_ID} to {zip_path}")
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=zip_path, quiet=False)
        
        # Unzip the file
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(CHROMA_PATH)
        
        # Clean up the zip file
        os.remove(zip_path)
        logger.info("✅ Database setup complete.")
        
    except Exception as e:
        logger.critical(f"❌ Critical error during database setup: {e}", exc_info=True)
        # Stop the application if DB setup fails
        raise RuntimeError("Could not set up the database.") from e

# Run the database setup before anything else
setup_database()


# --- 4. Model & Database Loading ---
try:
    logger.info("Initializing application components...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading sentence transformer model: {MODEL_NAME} onto device: '{device}'")
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("✅ Embedding model loaded successfully.")

    logger.info(f"Connecting to ChromaDB at path: {CHROMA_PATH}")
    db_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    logger.info(f"✅ Successfully connected to ChromaDB. Collection '{COLLECTION_NAME}' contains {collection.count()} items.")

except Exception as e:
    logger.critical(f"❌ Critical error during component initialization: {e}", exc_info=True)
    collection = None
    model = None


# --- 5. Pydantic Models for API I/O ---
class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="The full-text query for semantic search.", example="velvet kitchen rug")
    keywords: List[str] = Field(..., description="A list of keywords for initial filtering.", example=["rug", "velvet"])

class SearchResult(BaseModel):
    id: str = Field(..., description="Unique identifier of the product.")
    name: str = Field(..., description="The Persian name of the product.")
    score: float = Field(..., description="The semantic similarity score as a percentage (0-100).")


# --- 6. FastAPI Application ---
app = FastAPI(
    title="Hybrid Product Search API",
    description="An API that performs a two-stage hybrid search: keyword-based filtering followed by semantic re-ranking.",
    version=API_VERSION
)

@app.post("/hybrid-search/", response_model=List[SearchResult])
def hybrid_search(request: HybridSearchRequest):
    # The rest of the API code remains the same...
    logger.info(f"Received search request. Query: '{request.query}', Keywords: {request.keywords}")

    if collection is None or model is None:
        logger.error("Search aborted because a required component (DB or Model) is not available.")
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")

    if not request.keywords:
        logger.warning("Request received with empty keywords list.")
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
        logger.info("No results found after keyword filtering.")
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
    return { "status": "OK", "message": "Hybrid search server is running.", "version": API_VERSION }

