from __future__ import annotations
from typing import List, Iterable, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required. Install with `pip install sentence-transformers`."
    ) from e

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    """
    Minimal local embedder.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in np.asarray(emb)]

    def embed_text(self, text: str) -> List[float]:
        result = self.embed_texts([text])
        return result[0] if result else []

