import logging
import torch
import numpy as np
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer, util

# --- 1. Configuration & Constants ---
# This path points to the volume mount location inside the Docker container.
CHROMA_PATH = "/app/product_db"
COLLECTION_NAME = "products"
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
API_VERSION = "0.1.0"
TOP_K_RESULTS = 15

# --- 2. Logging Setup ---
# Configure logger to output to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Model & Database Loading ---
# This block runs once when the application starts.
try:
    logger.info("Initializing application components...")

    # Determine the appropriate device for torch (CUDA if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading sentence transformer model: {MODEL_NAME} onto device: '{device}'")
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("✅ Embedding model loaded successfully.")

    # Establish a persistent connection to the ChromaDB client
    logger.info(f"Connecting to ChromaDB at path: {CHROMA_PATH}")
    db_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    logger.info(f"✅ Successfully connected to ChromaDB. Collection '{COLLECTION_NAME}' contains {collection.count()} items.")

except Exception as e:
    logger.critical(f"❌ Critical error during initialization: {e}", exc_info=True)
    # Set collection to None to indicate failure, which will be handled in the API endpoint
    collection = None
    model = None

# --- 4. Pydantic Models for API I/O ---
class HybridSearchRequest(BaseModel):
    """Defines the expected input for the hybrid search endpoint."""
    query: str = Field(..., description="The full-text query for semantic search.", example="velvet kitchen rug")
    keywords: List[str] = Field(..., description="A list of keywords for initial filtering.", example=["rug", "velvet"])

class SearchResult(BaseModel):
    """Defines the structure of a single search result item."""
    id: str = Field(..., description="Unique identifier of the product.")
    name: str = Field(..., description="The Persian name of the product.")
    score: float = Field(..., description="The semantic similarity score as a percentage (0-100).")


# --- 5. FastAPI Application ---
app = FastAPI(
    title="Hybrid Product Search API",
    description="An API that performs a two-stage hybrid search: keyword-based filtering followed by semantic re-ranking.",
    version=API_VERSION
)

@app.post("/hybrid-search/", response_model=List[SearchResult])
def hybrid_search(request: HybridSearchRequest):
    """
    Performs a hybrid search by first filtering with keywords and then
    re-ranking the results based on semantic similarity to the full query.
    """
    logger.info(f"Received search request. Query: '{request.query}', Keywords: {request.keywords}")

    if collection is None or model is None:
        logger.error("Search aborted because a required component (DB or Model) is not available.")
        raise HTTPException(status_code=503, detail="Service unavailable: Database or model not loaded.")

    if not request.keywords:
        logger.warning("Request received with empty keywords list.")
        raise HTTPException(status_code=400, detail="Keywords list cannot be empty.")

    # --- Step 1: Keyword-based Filtering (Corrected Logic) ---
    try:
        where_filter = {}
        # Handle single vs. multiple keywords for ChromaDB filter
        if len(request.keywords) == 1:
            # If there's only one keyword, use a simple $contains filter
            where_filter = {"$contains": request.keywords[0]}
        else:
            # If there are multiple keywords, use the $or filter
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

    # --- Step 2: Semantic Re-ranking ---
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
    """
    Provides a simple health check endpoint to confirm the server is running.
    """
    return {
        "status": "OK",
        "message": "Hybrid search server is running.",
        "version": API_VERSION
    }

