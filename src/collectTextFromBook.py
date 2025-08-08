import os
import json
import logging
import requests
from io import BytesIO, StringIO
from urllib.parse import urlparse
from typing import List, Union
from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_huggingface import HuggingFaceEmbeddings
import docx
import pandas as pd
from bs4 import BeautifulSoup
import fitz

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixedFileTypeLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm", ".csv", ".json"
    }

    def __init__(self, file_paths: Union[str, List[str]], verbose: bool = True):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self.file_paths = self._gather_supported_files(file_paths)
        self.verbose = verbose
        if not self.verbose:
            logger.setLevel(logging.WARNING)

    def _gather_supported_files(self, paths: List[str]) -> List[str]:
        all_files = []
        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    all_files.extend([
                        os.path.join(root, file)
                        for file in files
                        if os.path.splitext(file)[1].lower() in self.SUPPORTED_EXTENSIONS
                    ])
            else:
                ext = os.path.splitext(path)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    all_files.append(path)
                else:
                    logger.warning(f"Unsupported file type: {path}")
        return all_files

    def load(self) -> List[Document]:
        docs = []
        futures = []
        with ThreadPoolExecutor() as executor:
            for path in self.file_paths:
                futures.append(executor.submit(self._load_single, path))
            for future in as_completed(futures):
                result = future.result()
                if result:
                    if isinstance(result, list):
                        docs.extend(result)
                    else:
                        docs.append(result)
        return docs

    def _load_single(self, path: str):
        try:
            is_url = path.startswith("http://") or path.startswith("https://")
            ext = os.path.splitext(urlparse(path).path if is_url else path)[1].lower()

            logger.info(f"Loading: {path}")
            loader_fn = {
                ".pdf": self._load_pdf,
                ".docx": self._load_word,
                ".doc": self._load_word,
                ".txt": self._load_txt,
                ".md": self._load_txt,
                ".html": self._load_html,
                ".htm": self._load_html,
                ".csv": self._load_csv,
                ".json": self._load_json
            }.get(ext)

            if loader_fn:
                f = self._get_file_buffer(path) if is_url else path
                result = loader_fn(f)
                if isinstance(result, list):
                    for r in result:
                        r.metadata["source"] = path
                    return result
                else:
                    return Document(page_content=result, metadata={"source": path})
            else:
                logger.warning(f"Unsupported extension: {ext}")
        except Exception as e:
            logger.error(f"Error loading {path}: {e}", exc_info=True)
        return None

    def _get_file_buffer(self, url: str) -> Union[BytesIO, StringIO]:
        logger.info(f"Fetching from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        is_binary = "application" in content_type or "octet-stream" in content_type
        return BytesIO(response.content) if is_binary else StringIO(response.text)

    def _load_pdf(self, file: Union[str, BytesIO]) -> str:
        """Uses PyMuPDF to load PDF text."""
        text = []
        doc = fitz.open(file) if isinstance(file, str) else fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            t = page.get_text()
            if t:
                text.append(t)
        return "\n".join(text)

    def _load_word(self, file: Union[str, BytesIO]) -> str:
        f = open(file, "rb") if isinstance(file, str) else file
        doc = docx.Document(f)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _load_txt(self, file: Union[str, StringIO]) -> str:
        return open(file, encoding="utf-8").read() if isinstance(file, str) else file.read()

    def _load_html(self, file: Union[str, StringIO]) -> str:
        raw = open(file, encoding="utf-8").read() if isinstance(file, str) else file.read()
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def _load_csv(self, file: Union[str, StringIO]) -> List[Document]:
        df = pd.read_csv(file)
        return [
            Document(
                page_content="\n".join(f"{col}: {row[col]}" for col in df.columns),
                metadata={"row": idx}
            )
            for idx, row in df.iterrows()
        ]

    def _load_json(self, file: Union[str, StringIO]) -> List[Document]:
        data = json.load(open(file, encoding="utf-8")) if isinstance(file, str) else json.load(file)
        return [Document(page_content=json.dumps(obj), metadata={}) for obj in data] if isinstance(data, list) else [Document(page_content=json.dumps(data), metadata={})]

# from utility import load_documents, split_documents, embed_chunks,build_embedding_json_for_db

# def main():
#     # 📂 Step 1: Define input source(s)
#     file_inputs = ["./docs"]  # Replace with your file paths or folder

#     # ⚙️ Step 2: Load documents
#     print("[*] Loading documents...")
#     docs = load_documents(file_inputs)

#     # 📏 Step 3: Split documents into manageable chunks
#     print(f"[*] Splitting {len(docs)} documents into chunks...")
#     chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

#     # 🤖 Step 4: Load embedding model
#     print("[*] Loading embedding model...")
#     model_name = "sentence-transformers/all-MiniLM-L12-v2"
#     embedding_model = HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs={"device": "cpu"},  # Use "cuda" for GPU
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     # 🔢 Step 5: Embed the chunks
#     print(f"[*] Embedding {len(chunks)} chunks...")
#     embeddings = embed_chunks(chunks, embedding_model)

#     # 🧱 Step 6: Build embedding data for DB
#     print("[*] Building embedding JSON for DB...")
#     embedding_json, doc_ids = build_embedding_json_for_db(
#         chunks, embeddings, embedding_model_name=model_name
#     )

#     print(f"[✅] Processed {len(embedding_json)} embeddings from {len(doc_ids)} documents.")
#     return embedding_json  # <-- Return instead of saving

# # Example usage
# if __name__ == "__main__":
#     output = main()
#     import json
#     print(json.dumps(output, indent=2, ensure_ascii=False))