import os
import json
import logging
from typing import List
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from langchain_huggingface import HuggingFaceEmbeddings
from runForEmbeding import get_vectors_and_details

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Load .env
# ----------------------------
load_dotenv()

# ----------------------------
# MongoDB Utility Functions
# ----------------------------
def insert_chunks_to_db(embedding_json, collection):
    operations = []
    for entry in embedding_json:
        operations.append(
            UpdateOne(
                {"unique_chunk_id": entry["unique_chunk_id"]},
                {"$setOnInsert": entry},
                upsert=True
            )
        )
    if not operations:
        return 0
    result = collection.bulk_write(operations, ordered=False)
    inserted_count = result.upserted_count
    logger.info(f"Bulk upserted {len(operations)} chunks. Inserted new: {inserted_count}, matched existing: {result.matched_count}")
    return inserted_count

def get_mongo_collection(db_name: str = None, collection_name: str = None):
    """Return a MongoDB collection object using provided or environment values."""
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        raise ValueError("Please set MONGODB_URI environment variable.")

    db_name = db_name or os.environ.get("DB_NAME")
    collection_name = collection_name or os.environ.get("COLLECTION_NAME")

    if not db_name or not collection_name:
        raise ValueError("DB name and Collection name must be provided either as arguments or in environment variables.")

    client = MongoClient(MONGODB_URI)
    logger.info(f"Connected to MongoDB: {db_name}.{collection_name}")
    return client[db_name][collection_name], collection_name

# ----------------------------
# Main Processing Function
# ----------------------------
def create_vector_and_store_in_atlas(file_inputs: List[str], db_name: str = None, collection_name: str = None, embedding_model=None):
    if embedding_model is None:
        logger.info("Loading HuggingFace embedding model...")
        embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Loaded model: {embedding_model_name}")
    else:
        embedding_model_name = getattr(embedding_model, 'model_name', 'provided_model')

    logger.info("Generating vectors and extracting metadata from files...")
    vector, doc_ids = get_vectors_and_details(file_inputs, embedding_model=embedding_model)
    logger.info(f"Generated embeddings for {len(vector)} chunks")

    collection, used_collection_name = get_mongo_collection(db_name, collection_name)

    logger.info("Inserting new chunks into MongoDB...")
    inserted_count = insert_chunks_to_db(vector, collection)
    logger.info(f"✅ Inserted {inserted_count} new chunks into MongoDB collection '{used_collection_name}'.")

    # Print a summary
    unique_file_names = list({e["file_name"] for e in vector})
    summary = {
        "num_chunks": len(vector),
        "inserted_chunks": inserted_count,
        "file_names": unique_file_names,
        "embedding_model": embedding_model_name,
        "all_unique_ids": doc_ids,
    }

    logger.info("Summary of operation:")
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
# ----------------------------
# Example usage (uncomment to run directly)
# ----------------------------
# if __name__ == "__main__":
#     files = ["doc1.txt", "doc2.txt"]
#     create_vector_and_store_in_atlas(files, db_name="myDatabase", collection_name="myCollection")


# ----------------------------
# Entry Point
# ----------------------------
# if __name__ == "__main__":
#     file_inputs = ["/Users/abhishek/Desktop/tutorAi/data/jesc1dd"]
#     create_vector_and_store_in_atlas(file_inputs)
