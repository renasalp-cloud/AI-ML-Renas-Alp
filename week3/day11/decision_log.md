# Decision Log (Day 11)

# Goal
Build a local document Q&A system (RAG) that can run without sending data to any cloud service. The system should retrieve relevant document sections and answer questions using a local LLM.

# Serving choice
I used **Ollama** because it was the fastest way to run a local model with an OpenAI-compatible API endpoint. This made it easy to test models and keep the code simple.

# Operating system choice
I developed on **Windows**  to reduce setup time and avoid WSL installation issues. The pipeline only needs a local API endpoint, so the same code can run on WSL later if needed.

# Document ingestion
- Input documents are PDF files placed under `documents/`.
- PDFs were loaded with a PDF loader and split into chunks.
- The split settings started with:
  - `CHUNK_SIZE = 512`
  - `CHUNK_OVERLAP = 50`
These values were selected as a safe baseline for mixed documents (policies, handbook, report, product guide, meeting notes).

# Embeddings choice
I used **nomic-embed-text** through Ollama for embeddings. This keeps the full pipeline local and avoids external embedding APIs.

# Vector store choice
I used **ChromaDB** because it is lightweight, simple to set up, and works well for local development. The vector database is stored in the `chroma_db/` folder.

# Retrieval settings
- `TOP_K = 5` for retrieval to capture enough context when answers are spread across multiple parts of the documents.
- In practice, sending too much context to the model sometimes produced unstable outputs, so I limited what is passed into generation:
  - `MAX_CONTEXT_CHUNKS = 3`
  - `MAX_CHUNK_CHARS = 1200`

# Model selection and swap
I initially tested **qwen3:4b** because it is small and runs on my hardware. However, in the full pipeline it often produced long “analysis-style” responses and sometimes returned empty output in the `content` field.

To improve answer stability and follow “final answer only” formatting better, I switched the LLM to **llama3.2:3b**. This change was done using configuration only (no code changes), which supports a clean model swap approach.

# Handling empty model output
During generation, the model sometimes returned empty output. I added a simple retry strategy:
- First call uses normal context.
- If the answer is empty, retry with a simpler prompt and only the top chunk.
- If still empty, return the standard fallback message:
  "I cannot find this information in the provided documents."

# Benchmarking approach
I created a small benchmark runner:
- Runs a set of test questions from `test_questions.json`
- Captures retrieval time, generation time, and total time
- Saves results to `benchmark_results.json`

The correctness check is intentionally simple and mainly used to identify which questions need better retrieval or clearer expectations.

# Notes / next improvements
- Improve test question expected answers (make them more specific and consistent).
- Add a better evaluation method (manual scoring or keyword-based scoring per question).
- Tune chunk size and top_k based on failure cases (especially for comparison questions and policy updates with dates).