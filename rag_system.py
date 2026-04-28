"""
RAG System — Orchestration Core
================================
Ties together parsing, chunking, embedding, vector storage, retrieval, and
LLM generation into a single high-level interface.
"""

from typing import Dict, Any, Optional, List

from db import (
    UploadRecord,
    ChatHistory,
    get_db_session,
)
from contract_parser import (
    extract_text_from_pdf_bytes,
    extract_text_from_s3,
    list_s3_objects,
)
from vector_store import (
    chunk_text,
    upsert_chunks,
    query_similar,
    delete_namespace,
    make_namespace,
)
from external_services import generate_answer


# ── PDF Upload Pipeline ──────────────────────────────────────────────────────

def process_pdf_upload(
    user_id: int,
    filename: str,
    pdf_bytes: bytes,
    llm_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: PDF bytes → text → chunks → embeddings → Pinecone.

    Returns:
        {
            "upload_id": int,
            "namespace": str,
            "chunk_count": int,
            "status": "ready" | "error",
            "message": str,
        }
    """
    db = get_db_session()
    try:
        # 1. Create upload record
        record = UploadRecord(
            user_id=user_id,
            source_type="pdf",
            filename=filename,
            status="processing",
            pinecone_namespace="pending",
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        namespace = make_namespace(user_id, record.id)
        record.pinecone_namespace = namespace
        db.commit()

        # 2. Extract text
        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text.strip():
            record.status = "error"
            db.commit()
            return {
                "upload_id": record.id,
                "namespace": namespace,
                "chunk_count": 0,
                "status": "error",
                "message": "No text could be extracted from the PDF.",
            }

        # 3. Chunk
        chunks = chunk_text(text)

        # 4. Embed & upsert
        count = upsert_chunks(
            chunks,
            namespace=namespace,
            source_metadata={"filename": filename, "source": "pdf"},
        )

        # 5. Finalise
        record.chunk_count = count
        record.status = "ready"
        db.commit()

        return {
            "upload_id": record.id,
            "namespace": namespace,
            "chunk_count": count,
            "status": "ready",
            "message": f"Successfully processed '{filename}' into {count} chunks.",
        }

    except Exception as e:
        db.rollback()
        return {
            "upload_id": getattr(record, "id", None),
            "namespace": "",
            "chunk_count": 0,
            "status": "error",
            "message": str(e),
        }
    finally:
        db.close()


# ── AWS S3 Pipeline ──────────────────────────────────────────────────────────

def process_s3_connection(
    user_id: int,
    s3_prefix: str = "",
    bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch files from S3, extract text, chunk, embed, and store.

    Returns:
        {"upload_id": int, "namespace": str, "chunk_count": int, ...}
    """
    db = get_db_session()
    try:
        keys = list_s3_objects(prefix=s3_prefix, bucket=bucket)
        if not keys:
            return {"status": "error", "message": "No objects found in S3."}

        record = UploadRecord(
            user_id=user_id,
            source_type="aws",
            s3_path=s3_prefix,
            status="processing",
            pinecone_namespace="pending",
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        namespace = make_namespace(user_id, record.id)
        record.pinecone_namespace = namespace
        db.commit()

        total_chunks = 0
        for key in keys:
            text = extract_text_from_s3(key, bucket=bucket)
            if text.startswith("[Unsupported"):
                continue
            chunks = chunk_text(text)
            count = upsert_chunks(
                chunks,
                namespace=namespace,
                source_metadata={"filename": key, "source": "aws_s3"},
            )
            total_chunks += count

        record.chunk_count = total_chunks
        record.status = "ready"
        db.commit()

        return {
            "upload_id": record.id,
            "namespace": namespace,
            "chunk_count": total_chunks,
            "status": "ready",
            "message": f"Processed {len(keys)} files into {total_chunks} chunks.",
        }

    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


# ── Question Answering ───────────────────────────────────────────────────────

def ask_question(
    user_id: int,
    upload_id: int,
    question: str,
    llm_name: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks and generate an answer.

    Returns:
        {
            "answer": str,
            "llm_used": str,
            "sources": [{"text": str, "score": float, ...}, ...],
        }
    """
    namespace = make_namespace(user_id, upload_id)

    # Retrieve
    matches = query_similar(query=question, namespace=namespace, top_k=top_k)
    if not matches:
        return {
            "answer": "I couldn't find relevant information in the uploaded document to answer your question.",
            "llm_used": "none",
            "sources": [],
        }

    # Build context
    context = "\n\n---\n\n".join(m["text"] for m in matches)

    # Generate
    result = generate_answer(context, question, llm_name=llm_name)

    # Persist to chat history
    db = get_db_session()
    try:
        history = ChatHistory(
            user_id=user_id,
            upload_id=upload_id,
            question=question,
            answer=result["answer"],
            llm_used=result["llm_used"],
        )
        db.add(history)
        db.commit()
    finally:
        db.close()

    return {
        "answer": result["answer"],
        "llm_used": result["llm_used"],
        "sources": matches,
    }


# ── Chat History ─────────────────────────────────────────────────────────────

def get_chat_history(user_id: int, upload_id: Optional[int] = None) -> List[Dict]:
    """Return previous Q&A pairs for a user (optionally filtered by upload)."""
    db = get_db_session()
    try:
        query = db.query(ChatHistory).filter(ChatHistory.user_id == user_id)
        if upload_id:
            query = query.filter(ChatHistory.upload_id == upload_id)
        rows = query.order_by(ChatHistory.created_at.asc()).all()

        return [
            {
                "question": r.question,
                "answer": r.answer,
                "llm_used": r.llm_used,
                "timestamp": r.created_at.isoformat(),
            }
            for r in rows
        ]
    finally:
        db.close()


# ── Cleanup ──────────────────────────────────────────────────────────────────

def delete_upload(user_id: int, upload_id: int) -> Dict[str, str]:
    """Delete an upload's vectors and database record."""
    namespace = make_namespace(user_id, upload_id)
    delete_namespace(namespace)

    db = get_db_session()
    try:
        db.query(UploadRecord).filter(UploadRecord.id == upload_id).delete()
        db.commit()
        return {"status": "deleted", "message": f"Upload {upload_id} removed."}
    finally:
        db.close()
