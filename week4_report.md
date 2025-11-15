# Week 4 Report – Embeddings and Vector Store

## 1. Overview
This week I built the core indexing and retrieval pipeline for my RAG project.  
The goal was to embed processed text data, store it in a vector database, and test retrieval.

## 2. Work Done
- Implemented **embed.py** using `sentence-transformers` (MiniLM-L6-v2).
- Created **vector_store.py** using **ChromaDB** for local vector storage.
- Built **ingest.py** to read `.txt` files from `data/processed` and store them as embeddings.
- Added a **CLI** interface:
  - `python -m index build --from data/processed --reset` – builds or resets the index
  - `python -m index query "question"` – runs a search query
- Added **retrieval_tests.ipynb** to test and visualize search results.

## 3. Results
Example queries and outputs:
- **Query:** “What is Chroma used for?”  
  → Chroma is a local vector database used to store and search embeddings.
- **Query:** “Which model is small and fast?”  
  → MiniLM is a small embedding model that is fast on CPU.

Index size: 6 chunks  
Search time: ~0.1 s per query  

Retrieval results were relevant and consistent with expectations.

## 4. Notes
- Embeddings are normalized for cosine similarity.  
- The modular design makes it easy to switch embedding models later.
- Everything runs fully offline (no API keys needed).

## 5. Next Steps
- Integrate with the LLM for full RAG (QA pipeline) in Week 5.
