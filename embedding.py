"""
Embedding Layer (Open-Source)
=============================
Uses sentence-transformers running 100% locally.
No API keys needed. Downloads the model on first run (~80 MB).
"""

from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> List[float]:
    """Embed a single text into a 384-dim vector."""
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a query string (same model)."""
    return embed_text(query)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts in one efficient batch."""
    model = _get_model()
    results = model.encode(texts, convert_to_numpy=True)
    return results.tolist()


def get_embedding_dimension() -> int:
    return EMBEDDING_DIMENSION
