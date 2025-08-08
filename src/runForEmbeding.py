from langchain_huggingface import HuggingFaceEmbeddings
from utility import (
    load_documents,
    split_documents,
    embed_chunks,
    build_embedding_json_for_db
)

def get_vectors_and_details(file_inputs, embedding_model=None):
    # 📂 Step 1: Define input source(s)
    # file_inputs is passed as parameter now

    # ⚙️ Step 2: Load documents
    print("[*] Loading documents...")
    docs = load_documents(file_inputs)

    # 📏 Step 3: Split documents into manageable chunks
    print(f"[*] Splitting {len(docs)} documents into chunks...")
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

    # 🤖 Step 4: Load embedding model if not provided
    if embedding_model is None:
        print("[*] Loading embedding model...")
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Use "cuda" for GPU
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        model_name = getattr(embedding_model, 'model_name', 'provided_model')

    # 🔢 Step 5: Embed the chunks
    print(f"[*] Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks, embedding_model)

    # 🧱 Step 6: Build embedding data for DB
    print("[*] Building embedding JSON for DB...")
    embedding_json, doc_ids = build_embedding_json_for_db(
        chunks, embeddings, embedding_model_name=model_name
    )

    print(f"[✅] Processed {len(embedding_json)} embeddings from {len(doc_ids)} documents.")
    return embedding_json,doc_ids

# # Example usage
# if __name__ == "__main__":
#     import json
#     file_inputs = [
#         "/Users/abhishek/Desktop/tutorAi/data/jemh1dd/jemh1a1.pdf",
#         "/Users/abhishek/Desktop/tutorAi/data/jemh1dd/jemh1a2.pdf"
#     ]
#     vector = get_vectors_and_details(file_inputs)
#     print(json.dumps(vector, indent=3, ensure_ascii=False))  # Just for visualization
