import os
import json
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from collectText import MixedFileTypeLoader
from pymongo import MongoClient
def load_documents(file_paths: List[str], verbose: bool = True) -> List[Document]:
    loader = MixedFileTypeLoader(file_paths, verbose=verbose)
    return loader.load()

def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def embed_chunks(chunks: List[Document], embedding_model) -> List[List[float]]:
    texts = [chunk.page_content for chunk in chunks]
    return embedding_model.embed_documents(texts)

def build_embedding_json(chunks: List[Document], embeddings: List[List[float]], embedding_model_name: str) -> List[dict]:
    result = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        source = chunk.metadata.get("source", None)
        file_type = os.path.splitext(source)[1] if source else None
        file_name = os.path.basename(source) if source else None
        unique_id = f"{file_name}_{idx}"
        entry = {
            "index": idx,
            # "source": source,
            "file_name": file_name,
            "file_type": file_type,
            "unique_id": unique_id,
            "embedding": emb,  # Store full embedding for DB
            "embedding_size": len(emb),
            "chunk_text": chunk.page_content,  # Store full chunk for DB
            "embedding_model": embedding_model_name,
            "metadata": chunk.metadata
        }
        result.append(entry)
    return result

def main():
    file_paths = [
        # "/Users/abhishek/Desktop/vectorSearch_with_monogdb/LLL_ChtSht.pdf",
        "/Users/abhishek/Desktop/vectorSearch_with_monogdb/howto-sockets.pdf",
    ]
    docs = load_documents(file_paths)
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

    embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    embeddings = embed_chunks(chunks, embedding_model)
    embedding_json = build_embedding_json(chunks, embeddings, embedding_model_name)

    # Store in MongoDB
    MONGODB_URI = "mongodb+srv://abhishek1233445:A0t24VdRZzQ0eJSa@cluster0.fgmkf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DB_NAME = "atlas_vector"
    COLLECTION_NAME = "a_vectors"
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    inserted_count = 0
    for entry in embedding_json:
        if not collection.find_one({"unique_id": entry["unique_id"]}):
            collection.insert_one(entry)
            inserted_count += 1
    print(f"✅ Inserted {inserted_count} new chunks into MongoDB collection '{COLLECTION_NAME}'.")

    # Print only the first 2 entries, with only 5 embedding values and 50 chunk chars
    def light_entry(e):
        return {
            k: (e[k][:5] if k == "embedding" else e[k][:50] if k == "chunk_text" else e[k])
            for k in e if k in ["index", "source", "file_name", "file_type", "unique_id", "embedding", "embedding_size", "chunk_text", "embedding_model", "metadata"]
        }
    print(json.dumps([light_entry(e) for e in embedding_json[:2]], indent=2, default=str))

if __name__ == "__main__":
    main()