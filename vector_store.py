"""
Vector Store Layer (Open-Source)
================================
Uses ChromaDB as a local vector database — no cloud service needed.
Manages collections, upserting chunks, querying, and namespace isolation.
"""

import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from embedding import embed_text, embed_texts, embed_query


# ── ChromaDB Client ──────────────────────────────────────────────────────────

_chroma_client = None


def _get_chroma() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_collection(namespace: str):
    """
    Get or create a ChromaDB collection for the given namespace.
    Each user upload gets its own collection (acts like a Pinecone namespace).
    """
    client = _get_chroma()
    collection_name = f"{CHROMA_COLLECTION_NAME}_{namespace}"
    # ChromaDB collection names: 3-63 chars, alphanumeric + underscores/hyphens
    collection_name = collection_name[:63]
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Break text into overlapping chunks.

    Chunking strategy (as specified):
        Chunk 0: characters [0   .. 499]
        Chunk 1: characters [400 .. 899]
        Chunk 2: characters [800 .. 1299]
        ...

    Each chunk dict contains:
        - text:        the chunk content
        - start_char:  starting character index
        - end_char:    ending character index
        - chunk_index: sequential chunk number (0-based)
    """
    chunks: List[Dict[str, Any]] = []
    step = chunk_size - overlap  # 500 - 100 = 400
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_content = text[start:end]

        if chunk_content.strip():  # skip empty chunks
            chunks.append({
                "text": chunk_content,
                "start_char": start,
                "end_char": end,
                "chunk_index": idx,
            })
            idx += 1

        start += step

    return chunks


# ── Upsert ───────────────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: List[Dict[str, Any]],
    namespace: str,
    source_metadata: Optional[Dict[str, str]] = None,
    batch_size: int = 100,
) -> int:
    """
    Embed and upsert text chunks into ChromaDB.

    Args:
        chunks:          Output of chunk_text().
        namespace:       Collection namespace (typically user_id + upload_id).
        source_metadata: Extra metadata attached to every vector (e.g. filename).
        batch_size:      Documents per upsert call.

    Returns:
        Total number of vectors upserted.
    """
    collection = get_collection(namespace)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        ids = [str(uuid.uuid4()) for _ in batch_chunks]
        metadatas = []
        for chunk in batch_chunks:
            meta = {
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "chunk_index": chunk["chunk_index"],
            }
            if source_metadata:
                meta.update(source_metadata)
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=metadatas,
        )
        total += len(batch_chunks)

    return total


# ── Query ────────────────────────────────────────────────────────────────────

def query_similar(
    query: str,
    namespace: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Find the most relevant chunks for a user query.

    Args:
        query:           Natural-language question.
        namespace:       Collection namespace to search within.
        top_k:           Maximum results to return.
        score_threshold: Minimum similarity to include (ChromaDB returns
                         distances; lower = more similar for cosine).

    Returns:
        List of dicts with 'text', 'score', and other metadata.
    """
    collection = get_collection(namespace)

    # Check if collection has documents
    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    if results and results["documents"] and results["documents"][0]:
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            similarity = 1 - (distance / 2)
            if similarity >= score_threshold:
                matches.append({
                    "text": doc,
                    "score": round(similarity, 4),
                    "chunk_index": meta.get("chunk_index"),
                    "start_char": meta.get("start_char"),
                    "end_char": meta.get("end_char"),
                })

    return matches


# ── Namespace Management ─────────────────────────────────────────────────────

def delete_namespace(namespace: str) -> None:
    """Remove a collection (namespace) entirely."""
    client = _get_chroma()
    collection_name = f"{CHROMA_COLLECTION_NAME}_{namespace}"[:63]
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection may not exist


def make_namespace(user_id: int, upload_id: int) -> str:
    """Generate a deterministic namespace string for a user's upload."""
    return f"user_{user_id}_upload_{upload_id}"
