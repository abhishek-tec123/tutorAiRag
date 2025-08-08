from typing import List, Union
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import os

def load_documents(file_paths: Union[str, List[str]], verbose: bool = True) -> List[Document]:
    from collectTextFromBook import MixedFileTypeLoader
    loader = MixedFileTypeLoader(file_paths, verbose=verbose)
    return loader.load()


def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def embed_chunks(chunks: List[Document], embedding_model) -> List[List[float]]:
    texts = [chunk.page_content for chunk in chunks]
    return embedding_model.embed_documents(texts)


# def build_embedding_json(chunks: List[Document], embeddings: List[List[float]]) -> List[dict]:
#     result = []
#     for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
#         entry = {
#             "index": idx,
#             "source": chunk.metadata.get("source", None),
#             "embedding": emb,
#             "embedding_size": len(emb),
#             "chunk_text": chunk.page_content,
#             "metadata": chunk.metadata
#         }
#         result.append(entry)
#     return result

def generate_custom_id(file_name, length=5):
    # Deterministically generate a 5-char alphanumeric string from the file name
    hash_digest = hashlib.sha256(file_name.encode('utf-8')).hexdigest()
    return hash_digest[:length]

def build_embedding_json_for_db(chunks: List[Document], embeddings: List[List[float]], embedding_model_name: str):
    result = []
    unique_ids = set()
    # Assign a unique_id (5-char custom alphanumeric) for each document (by file_name)
    file_name_to_unique_id = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", None)
        file_name = os.path.basename(source) if source else None
        if file_name and file_name not in file_name_to_unique_id:
            file_name_to_unique_id[file_name] = generate_custom_id(file_name, 5)
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        source = chunk.metadata.get("source", None)
        file_type = os.path.splitext(source)[1] if source else None
        file_name = os.path.basename(source) if source else None
        unique_id = file_name_to_unique_id.get(file_name, None)
        unique_ids.add(unique_id)
        unique_chunk_id = f"{unique_id}_{idx}"
        entry = {
            "file_name": file_name,
            "file_type": file_type,
            "unique_id": unique_id,  # Document-level unique id (5-char custom)
            "unique_chunk_id": unique_chunk_id,  # Chunk-level unique id
            "embedding": emb,  # Store full embedding for DB
            "embedding_size": len(emb),
            "chunk_text": chunk.page_content,  # Store full chunk for DB
            "embedding_model": embedding_model_name,
            "metadata": chunk.metadata
        }
        result.append(entry)
    return result, list(unique_ids)