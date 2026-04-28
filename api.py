"""
API / Orchestration Layer
=========================
FastAPI application exposing REST endpoints for authentication,
file upload, S3 connection, and question answering.
"""

from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from pydantic import BaseModel

from db import init_db
from rest_user import (
    UserCreate,
    UserLogin,
    authenticate_user,
    register_user,
    decode_access_token,
)
from rag_system import (
    process_pdf_upload,
    process_s3_connection,
    ask_question,
    get_chat_history,
    delete_upload,
)

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API with multi-LLM support",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    init_db()


# ── Auth Dependency ───────────────────────────────────────────────────────────

def get_current_user(authorization: str = Header(...)):
    """Extract and validate the Bearer token from the Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = authorization.split(" ", 1)[1]
    user = decode_access_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


# ── Request / Response Models ─────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    upload_id: int
    question: str
    llm_name: Optional[str] = None
    top_k: int = 5


class S3ConnectRequest(BaseModel):
    s3_prefix: str = ""
    bucket: Optional[str] = None


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register")
def api_register(data: UserCreate):
    result = register_user(data)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/auth/login")
def api_login(data: UserLogin):
    result = authenticate_user(data.username, data.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return result


# ── Upload Endpoints ──────────────────────────────────────────────────────────

@app.post("/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """Upload a PDF file for processing and indexing."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    result = process_pdf_upload(
        user_id=user.user_id,
        filename=file.filename,
        pdf_bytes=contents,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])
    return result


@app.post("/upload/s3")
def connect_s3(
    data: S3ConnectRequest,
    user=Depends(get_current_user),
):
    """Connect to an AWS S3 bucket and index its contents."""
    result = process_s3_connection(
        user_id=user.user_id,
        s3_prefix=data.s3_prefix,
        bucket=data.bucket,
    )
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])
    return result


# ── Question Endpoint ─────────────────────────────────────────────────────────

@app.post("/ask")
def ask(
    data: QuestionRequest,
    user=Depends(get_current_user),
):
    """Ask a question against an uploaded document."""
    return ask_question(
        user_id=user.user_id,
        upload_id=data.upload_id,
        question=data.question,
        llm_name=data.llm_name,
        top_k=data.top_k,
    )


# ── History Endpoint ──────────────────────────────────────────────────────────

@app.get("/history")
def history(
    upload_id: Optional[int] = None,
    user=Depends(get_current_user),
):
    """Retrieve chat history for the authenticated user."""
    return get_chat_history(user_id=user.user_id, upload_id=upload_id)


# ── Delete Endpoint ──────────────────────────────────────────────────────────

@app.delete("/upload/{upload_id}")
def remove_upload(
    upload_id: int,
    user=Depends(get_current_user),
):
    """Delete an upload and its associated vectors."""
    return delete_upload(user_id=user.user_id, upload_id=upload_id)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
