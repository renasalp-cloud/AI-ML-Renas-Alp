import time
from typing import Dict, Any, List

from langchain_community.vectorstores import Chroma

from config import (
    COLLECTION_NAME,
    TOP_K,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
)
from openai import OpenAI


class OllamaEmbeddings:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


def retrieve_chunks(query: str) -> Dict[str, Any]:
    t0 = time.time()

    embedding = OllamaEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        api_key="ollama",
        model=EMBEDDING_MODEL,
    )

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory="chroma_db",
        embedding_function=embedding,
    )

    results = db.similarity_search_with_score(query, k=TOP_K)

    retrieved = []
    for doc, score in results:
        meta = doc.metadata or {}
        retrieved.append(
            {
                "content": doc.page_content,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "relevance_score": float(score),
            }
        )

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "retrieval_time_ms": elapsed_ms,
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL,
    }


if __name__ == "__main__":
    q = "What is the maximum number of remote work days per week?"
    out = retrieve_chunks(q)
    print(f"Query: {out['query']}")
    print(f"Retrieval time (ms): {out['retrieval_time_ms']}")
    print(f"Chunks returned: {len(out['retrieved_chunks'])}")
    if out["retrieved_chunks"]:
        first = out["retrieved_chunks"][0]
        print("Top chunk source:", first["source"], "page:", first["page"])
        print(first["content"][:300])