"""
RAG System Configuration
========================
Central configuration — fully open-source stack.
Embeddings: sentence-transformers (local)
Vector DB:  ChromaDB (local)
LLM:        Ollama + Qwen (local)
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_app.db")

# ── Embeddings (local sentence-transformers) ─────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension

# ── ChromaDB (local vector store) ────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")

# ── Ollama (local LLM) ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")

# Default LLM to use: "qwen", "gemini" (optional), or "ollama"
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "qwen")

# ── Optional paid APIs (leave blank if not using) ────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── AWS (Optional data connection) ───────────────────────────────────────────
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")

# ── Chunking Settings ────────────────────────────────────────────────────────
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlapping characters between consecutive chunks

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ── Streamlit ─────────────────────────────────────────────────────────────────
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
