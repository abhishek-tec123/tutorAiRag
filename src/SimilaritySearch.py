import logging
import os
import numpy as np
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from langchain_huggingface import HuggingFaceEmbeddings
from structured_response import generate_response_from_groq

# -----------------------------
# Disable tokenizers parallelism warning
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Default Configuration
# -----------------------------
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb+srv://...")  # Masked here for safety
DEFAULT_DB_NAME = os.environ.get("DB_NAME", "class10")
DEFAULT_COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "math")
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Embedding
# -----------------------------
def embed_query(query: str, embedding_model) -> list:
    return embedding_model.embed_query(query)

# -----------------------------
# Vector Search (Atlas)
# -----------------------------
def find_similar_chunks(query_embedding, collection, num_candidates=100, limit=3, index_name=VECTOR_INDEX_NAME) -> list:
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": limit
            }
        },
        {
            "$project": {
                "chunk_text": 1,
                "score": {"$meta": "vectorSearchScore"},
                "unique_id": 1,
                "unique_chunk_id": 1,
            }
        }
    ]
    return list(collection.aggregate(pipeline))

# -----------------------------
# Vector Search (Fallback - In-Memory)
# -----------------------------
def find_similar_chunks_in_memory(query_embedding, collection, top_k=3):
    docs = list(collection.find({}, {"embedding": 1, "chunk_text": 1, "unique_id": 1, "unique_chunk_id": 1}))
    
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored = [
        {
            "unique_id": doc["unique_id"],
            "unique_chunk_id": doc.get("unique_chunk_id", "N/A"),
            "chunk_text": doc["chunk_text"],
            "score": cosine_similarity(query_embedding, doc["embedding"])
        }
        for doc in docs if "embedding" in doc
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# -----------------------------
# Main Execution
# -----------------------------
def retrieve_and_generate_llm_response(query: str, db_name: str = None, collection_name: str = None, embedding_model=None):
    db_name = db_name or DEFAULT_DB_NAME
    collection_name = collection_name or DEFAULT_COLLECTION_NAME

    logger.info(f"[*] Connecting to MongoDB: {db_name}.{collection_name}...")
    client = MongoClient(MONGODB_URI)
    collection = client[db_name][collection_name]

    if embedding_model is None:
        logger.info("[*] Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    logger.info("[*] Generating embedding for query...")
    query_embedding = embed_query(query, embedding_model)

    results = []
    try:
        logger.info("[*] Trying MongoDB Atlas vector search...")
        results = find_similar_chunks(query_embedding, collection)
        logger.info(f"[+] Retrieved {len(results)} chunks from Atlas vector index.")

        if not results:
            logger.warning("[!] No results from Atlas. Falling back to in-memory similarity search...")
            results = find_similar_chunks_in_memory(query_embedding, collection)
            logger.info(f"[+] Retrieved {len(results)} chunks from in-memory similarity.")
    except Exception as e:
        logger.warning(f"[!] Atlas vector search failed: {e}")
        logger.info("[*] Falling back to in-memory similarity search...")
        results = find_similar_chunks_in_memory(query_embedding, collection)
        logger.info(f"[+] Retrieved {len(results)} chunks from in-memory similarity.")

    if not results:
        logger.warning("[!] No similar documents found.")
        return

    print("--------------------------------------------------------- \n")
    for idx, doc in enumerate(results):
        logger.info(f"[{idx + 1}] Score: {doc['score']:.4f}")
        logger.info(f"Unique ID: {doc.get('unique_id', 'N/A')}")
        logger.info(f"Unique Chunk ID: {doc.get('unique_chunk_id', 'N/A')}")
        logger.info("---")

    result_string = "\n---\n".join(doc["chunk_text"] for doc in results if "chunk_text" in doc)

    logger.info("[*] Sending top chunks to LLM...")
    response_from_groq = generate_response_from_groq(input_text=result_string, query=query)

    # print("\n--- LLM Response ---")
    # print(response_from_groq)
    return response_from_groq

# -----------------------------
# Entry Point (optional)
# -----------------------------
# if __name__ == "__main__":
#     query = "Explain Newton's laws of motion"
#     retrieve_and_generate_llm_response(query, db_name="class10", collection_name="science")
