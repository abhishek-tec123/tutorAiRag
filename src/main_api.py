from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from VectorStoreInAtls import create_vector_and_store_in_atlas
from SimilaritySearch import retrieve_and_generate_llm_response
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI()
embedding_model = None  # will be set on startup

# -----------------------------
# Request Models
# -----------------------------
class VectorRequest(BaseModel):
    file_paths: List[str]
    class_: str  # use class_ to avoid Python keyword conflict
    subject: str

class SearchRequest(BaseModel):
    query: str
    class_: str
    subject: str

# -----------------------------
# Load model at startup
# -----------------------------
@app.on_event("startup")
def load_model_on_startup():
    global embedding_model
    logger.info("[*] Loading embedding model on startup...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("[+] Embedding model loaded and ready.")

# -----------------------------
# Utility: Map class & subject
# -----------------------------
def map_to_db_and_collection(class_: str, subject: str):
    db_name = class_.strip().lower()
    collection_name = subject.strip().lower()
    return db_name, collection_name

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/create_vectors")
def create_vectors(request: VectorRequest):
    try:
        db_name, collection_name = map_to_db_and_collection(request.class_, request.subject)

        summary = create_vector_and_store_in_atlas(
            file_inputs=request.file_paths,
            db_name=db_name,
            collection_name=collection_name,
            embedding_model=embedding_model
        )

        return {
            "status": "success",
            "message": f"Vectors created and stored in MongoDB Atlas → {db_name}.{collection_name}",
            "summary": summary
        }

    except Exception as e:
        logger.error(f"[!] Error in /create_vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_and_respond(request: SearchRequest):
    try:
        db_name, collection_name = map_to_db_and_collection(request.class_, request.subject)

        response = retrieve_and_generate_llm_response(
            query=request.query,
            db_name=db_name,
            collection_name=collection_name,
            embedding_model=embedding_model
        )

        return {
            "status": "success",
            "response": response
        }

    except Exception as e:
        logger.error(f"[!] Error in /search: {e}")
        raise HTTPException(status_code=500, detail=str(e))