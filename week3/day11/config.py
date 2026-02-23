import os

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")    # Default LLM model (can be changed via environment variable)
                                                     # Initially tested with qwen3:4b, switched to llama3.2:3b for more stable final answers

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
COLLECTION_NAME = "bootcamp_docs"

MAX_CONTEXT_CHUNKS = 3
MAX_CHUNK_CHARS = 1200

SYSTEM_PROMPT = """
You are a question answering system.

Answer the question using ONLY the provided context.

Rules:
- Do NOT explain your reasoning.
- Do NOT analyze the documents.
- Do NOT say "Let me analyze".
- Provide ONLY the final answer.
- Maximum 3 sentences.

If the answer is not in the context, say:
"I cannot find this information in the provided documents."

Context:
{context}
"""

MAX_TOKENS = 256
TEMPERATURE = 0.2