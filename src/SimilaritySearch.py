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
MONGODB_URI = os.environ.get("MONGODB_URI")
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

    logger.info(f"[*] Processing query: '{query}' for {db_name}.{collection_name}...")
    
    # Check if GROQ API key is available
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("[!] GROQ_API_KEY environment variable is not set.")
        return "GROQ API key is not configured. Please set the GROQ_API_KEY environment variable."
    
    try:
        logger.info(f"[*] Connecting to MongoDB: {db_name}.{collection_name}...")
        client = MongoClient(MONGODB_URI)
        
        # Check if database exists
        if db_name not in client.list_database_names():
            logger.error(f"[!] Database '{db_name}' does not exist.")
            return f"Database '{db_name}' does not exist. Please create vectors for this class first."
        
        # Check if collection exists
        if collection_name not in client[db_name].list_collection_names():
            logger.error(f"[!] Collection '{collection_name}' does not exist in database '{db_name}'.")
            return f"Collection '{collection_name}' does not exist in database '{db_name}'. Please create vectors for this subject first."
        
        collection = client[db_name][collection_name]
        
        # Check if collection has documents
        doc_count = collection.count_documents({})
        if doc_count == 0:
            logger.error(f"[!] Collection '{collection_name}' is empty.")
            return f"Collection '{collection_name}' is empty. Please create vectors for this subject first."
        
        logger.info(f"[+] Connected to MongoDB: {db_name}.{collection_name} (contains {doc_count} documents)")
    except OperationFailure as e:
        logger.error(f"[!] MongoDB connection failed: {e}")
        return "Failed to connect to MongoDB. Please check your connection string and credentials."
    except Exception as e:
        logger.error(f"[!] An unexpected error occurred during MongoDB connection: {e}")
        return "An unexpected error occurred during MongoDB connection."

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
        try:
            results = find_similar_chunks_in_memory(query_embedding, collection)
            logger.info(f"[+] Retrieved {len(results)} chunks from in-memory similarity.")
        except Exception as fallback_error:
            logger.error(f"[!] In-memory similarity search also failed: {fallback_error}")
            return "Failed to perform similarity search. Please check your database and try again."

    if not results:
        logger.warning("[!] No similar documents found.")
        return "No similar documents found for the query. Please ensure that vectors have been created for this class and subject."

    print("--------------------------------------------------------- \n")
    for idx, doc in enumerate(results):
        logger.info(f"[{idx + 1}] Score: {doc['score']:.4f}")
        logger.info(f"Unique ID: {doc.get('unique_id', 'N/A')}")
        logger.info(f"Unique Chunk ID: {doc.get('unique_chunk_id', 'N/A')}")
        logger.info("---")

    result_string = "\n---\n".join(doc["chunk_text"] for doc in results if "chunk_text" in doc)

    if not result_string.strip():
        logger.warning("[!] No text content found in results.")
        return "No text content found in the retrieved documents."

    logger.info("[*] Sending top chunks to LLM...")
    try:
        response_from_groq = generate_response_from_groq(input_text=result_string, query=query)
        if response_from_groq:
            logger.info("[+] LLM response generated successfully.")
            return response_from_groq
        else:
            logger.warning("[!] LLM returned empty response.")
            return "The AI model returned an empty response. Please try again."
    except Exception as e:
        logger.error(f"[!] Error generating LLM response: {e}")
        return f"Failed to generate AI response: {str(e)}"

# -----------------------------
# Entry Point (optional)
# -----------------------------
# if __name__ == "__main__":
#     query = "Explain Newton's laws of motion"
#     retrieve_and_generate_llm_response(query, db_name="class10", collection_name="science")
