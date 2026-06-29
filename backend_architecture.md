# Production RAG Backend Architecture

This document serves as a comprehensive technical reference for the RAG (Retrieval-Augmented Generation) backend. The backend is designed as a modular, production-ready system utilizing FastAPI, LangChain, ChromaDB, and Groq.

## 1. System Overview

The system strictly adheres to the principle of Separation of Concerns by isolating the **Offline Indexing Pipeline** (ingestion) from the **Online Query Pipeline** (serving requests).

### Tech Stack
*   **Web Framework:** FastAPI (with Uvicorn server)
*   **Orchestration:** LangChain (`langchain-core`, `langchain-chroma`, `langchain-huggingface`, `langchain-community`, `langchain-classic`)
*   **LLM Provider:** Groq (`llama-3.1-8b-instant`) for high-speed inference.
*   **Vector Database:** ChromaDB (Persistent local storage)
*   **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
*   **Package Manager:** `uv`

---

## 2. Core Modules

### `config.py` (Configuration)
Serves as the global configuration hub.
*   **Environment Variables:** Loads `.env` via `python-dotenv` (requires `GROQ_API_KEY`, `LANGSMITH_API_KEY`, etc.).
*   **LLM Initialization:** Exposes `get_llm()` which returns the `ChatGroq` instance initialized with `temperature=0` for deterministic outputs.
*   **Caching:** Implements an `InMemoryCache` using `langchain.globals.set_llm_cache()`. Exact duplicate LLM queries are instantly served from memory, costing 0 tokens.

### `database.py` (Vector Store & Indexing)
Manages the connection to data persistence layers.
*   **`get_or_create_vector_db()`:** Connects to or initializes the ChromaDB instance at `./chroma_db` using the `rag_collection` namespace and HuggingFace embeddings.
*   **`create_bm25_retriever()`:** A critical function for Hybrid Search. Because BM25 (keyword search) is a sparse, term-frequency index, it cannot be incrementally updated like a Vector DB. This function dynamically pulls all current text from ChromaDB and rebuilds the BM25 index entirely in memory upon server startup.

### `ingest.py` (Offline Indexing Pipeline)
A standalone script (`uv run ingest.py`) responsible for processing and storing documents without blocking the web server.
*   **Incremental Updates:** Uses `glob` to scan `/txt` and `/pdfs` folders. Compares file paths against the `source` metadata of chunks already existing in ChromaDB. Only new, unprocessed files are ingested.
*   **Dynamic Chunking Routing:** 
    *   `.txt` files → `CharacterTextSplitter` (chunk_size: 1000, overlap: 200, sep: `\n`)
    *   `.pdf` files → `TokenTextSplitter` (chunk_size: 250, overlap: 50)
    *   `.md` files → `MarkdownTextSplitter`
    *   Fallback → `RecursiveCharacterTextSplitter`
*   After chunking, files are embedded and added to the Chroma instance.

### `retriever.py` (Advanced RAG Pipeline)
Houses the complex retrieval logic and chain execution.
1.  **Base Retrievers:** Vector Search (Semantic) and BM25 (Keyword).
2.  **EnsembleRetriever:** Combines the two base retrievers using Reciprocal Rank Fusion (weights: `[0.5, 0.5]`). This guarantees both semantic meaning and exact keyword matches are captured.
3.  **MultiQueryRetriever:** A pre-retrieval optimization step. It uses the LLM to generate 3 semantically varying versions of the user's original query. The EnsembleRetriever is then executed for *all* variations in parallel to drastically improve recall.
4.  **Note on Contextual Compression:** An `LLMChainExtractor` (Post-retrieval optimization) was originally implemented but was temporarily disabled to adhere to Groq's Free Tier Token Per Minute (TPM) limit of 6,000. Compressing chunks individually causes massive parallel LLM API calls resulting in HTTP 429 Rate Limit errors.

### `main.py` (Online Query Pipeline / Web API)
The entry point for the FastAPI server.
*   Instantiates the LLM, connects to ChromaDB, and builds the BM25 index on startup.
*   Implements `CORSMiddleware` allowing `*` origins (Note: Must be restricted to the frontend URL in a deployed production environment).
*   Exposes `POST /api/chat` taking a `ChatRequest` Pydantic model (`{"message": "..."}`) and returning a `ChatResponse` (`{"answer": "..."}`).

---

## 3. LangSmith Observability
The backend relies heavily on LangSmith for tracking and debugging.
Because `LANGSMITH_TRACING=true` is set in the environment, every HTTP request hitting `/api/chat` generates a full trace.
You can monitor:
1.  The multi-query generation (what 3 queries the LLM came up with).
2.  The exact documents retrieved by the Hybrid search.
3.  The final prompt injected into the LLM.
4.  Latency and token consumption for every stage of the chain.

## 4. Running the Project

**1. To process new documents (Offline):**
```bash
cd rag
uv run ingest.py
```

**2. To run the API Server (Online):**
```bash
cd rag
uv run uvicorn main:app --reload
```
The server will run on `http://127.0.0.1:8000`. You can view interactive Swagger UI API documentation at `http://127.0.0.1:8000/docs`.
