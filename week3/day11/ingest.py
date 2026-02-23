import os
import time
from typing import List

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
)


class OllamaEmbeddings:
    """
    Minimal embeddings wrapper for Ollama's OpenAI-compatible /v1/embeddings endpoint.
    Sends raw text (string / list of strings). No token splitting.
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


def load_pdfs(folder_path: str):
    docs = []
    for name in os.listdir(folder_path):
        if name.lower().endswith(".pdf"):
            path = os.path.join(folder_path, name)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs


def main():
    documents_dir = "documents"
    persist_dir = "chroma_db"

    if not os.path.isdir(documents_dir):
        raise FileNotFoundError(f"Missing folder: {documents_dir}")

    print("Loading PDFs...")
    docs = load_pdfs(documents_dir)
    print(f"Loaded pages: {len(docs)}")

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks: {len(chunks)}")

    print("Creating embeddings and storing in ChromaDB...")
    t0 = time.time()

    embedding = OllamaEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        api_key="ollama",
        model=EMBEDDING_MODEL,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    vectorstore.persist()

    elapsed = round((time.time() - t0) * 1000)
    print(f"Done! Stored in '{persist_dir}' ({elapsed} ms).")


if __name__ == "__main__":
    main()