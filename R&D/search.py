import numpy as np
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from pymongo.errors import OperationFailure

def embed_query(query, embedding_model):
    return embedding_model.embed_query(query)

def find_similar_chunks_in_memory(query_embedding, collection, top_k=3):
    docs = list(collection.find({}, {"embedding": 1, "chunk_text": 1, "unique_id": 1}))
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    scored = [
        {
            "unique_id": doc["unique_id"],
            "chunk_text": doc["chunk_text"],
            "score": cosine_similarity(query_embedding, doc["embedding"])
        }
        for doc in docs
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def find_similar_chunks_atlas_vector_search(query_embedding, collection, num_candidates=100, limit=3, index_name="vector_index"):
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
                "unique_id": 1
            }
        }
    ]
    return list(collection.aggregate(pipeline))

def print_atlas_index_status(collection):
    try:
        logging.info("--- MongoDB Atlas Search Index Status ---")
        for idx in collection.list_search_indexes():
            logging.info(idx)
    except OperationFailure as e:
        logging.error(f"Could not retrieve index status: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    MONGODB_URI = "mongodb+srv://abhishek1233445:A0t24VdRZzQ0eJSa@cluster0.fgmkf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DB_NAME = "atlas_vector"
    COLLECTION_NAME = "a_vectors"
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    query = input("Enter your search query: ")
    query_emb = embed_query(query, embedding_model)

    print_atlas_index_status(collection)

    # print("\n--- In-memory Cosine Similarity Search ---")
    # top_chunks = find_similar_chunks_in_memory(query_emb, collection, top_k=5)
    # for chunk in top_chunks:
    #     print(f"Score: {chunk['score']:.4f}\nText: {chunk['chunk_text']}\n---")

    print("\n--- MongoDB Atlas Vector Search (Server-Side) ---")
    try:
        results = find_similar_chunks_atlas_vector_search(query_emb, collection, num_candidates=100, limit=3, index_name="vector_index")
        for doc in results:
            logging.info(f"Score: {doc['score']:.4f}\nText: {doc['chunk_text']}\n---")
    except Exception as e:
        logging.error(f"Atlas Vector Search failed: {e}")