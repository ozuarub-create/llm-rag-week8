# index/vector_store.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction

from .embed import Embedder


class _EmbedderEF(EmbeddingFunction):
    """
    Adapter that lets us plug our Embedder into Chroma.
    Chroma calls __call__(list[str]) -> list[list[float]]
    """
    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed_texts(input)


class VectorStore:
    """
    Minimal wrapper around Chroma persistent DB.
    Stores texts with optional metadata and lets you run similarity search.
    """

    def __init__(
        self,
        path: str = ".chroma",
        collection_name: str = "docs",
        embedder: Optional[Embedder] = None,
    ):
        os.makedirs(path, exist_ok=True)
        self.client: PersistentClient = chromadb.PersistentClient(path=path)
        self.embedding_fn = _EmbedderEF(embedder)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # cosine distance
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if not texts:
            return
        n = len(texts)
        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        if ids is None:
            # simple predictable ids; in real use ensure global uniqueness
            ids = [f"doc-{i}" for i in range(self.count(), self.count() + n)]

        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def count(self) -> int:
        return self.collection.count()

    def similarity_search(
        self, query: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Returns list of {text, metadata, distance} sorted by nearest.
        For cosine, smaller distance = more similar.
        """
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out: List[Dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            out.append({"text": text, "metadata": meta or {}, "distance": float(dist)})
        return out

