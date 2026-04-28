"""
Data Processing Layer — Contract / PDF Parser
==============================================
Extracts text from uploaded PDFs and from AWS S3 objects,
then prepares the content for chunking and embedding.
"""

import io
import tempfile
from typing import Optional

import PyPDF2
import pdfplumber
import boto3

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_S3_BUCKET,
)


# ── PDF Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract all text from in-memory PDF bytes.

    Uses pdfplumber first (better for tables / structured text).
    Falls back to PyPDF2 if pdfplumber yields nothing.
    """
    text = _extract_with_pdfplumber(pdf_bytes)
    if not text.strip():
        text = _extract_with_pypdf2(pdf_bytes)
    return clean_text(text)


def extract_text_from_pdf_path(file_path: str) -> str:
    """Extract text from a PDF file on disk."""
    with open(file_path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Use pdfplumber for extraction (handles tables well)."""
    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
    return "\n\n".join(pages_text)


def _extract_with_pypdf2(pdf_bytes: bytes) -> str:
    """Fallback extractor using PyPDF2."""
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)
    return "\n\n".join(pages_text)


# ── AWS S3 Extraction ────────────────────────────────────────────────────────

def get_s3_client():
    """Return a boto3 S3 client using configured credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def list_s3_objects(prefix: str = "", bucket: Optional[str] = None) -> list:
    """List objects in the configured S3 bucket under a given prefix."""
    s3 = get_s3_client()
    bucket = bucket or AWS_S3_BUCKET
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]


def extract_text_from_s3(s3_key: str, bucket: Optional[str] = None) -> str:
    """
    Download a file from S3 and extract its text.

    Supports:
        .pdf  → PDF extraction pipeline
        .txt  → direct read
        .csv  → direct read
    """
    s3 = get_s3_client()
    bucket = bucket or AWS_S3_BUCKET

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3.download_fileobj(bucket, s3_key, tmp)
        tmp_path = tmp.name

    lower_key = s3_key.lower()
    if lower_key.endswith(".pdf"):
        return extract_text_from_pdf_path(tmp_path)
    elif lower_key.endswith((".txt", ".csv", ".json", ".md")):
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            return clean_text(f.read())
    else:
        return f"[Unsupported file type: {s3_key}]"


# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise extracted text:
      - collapse multiple blank lines
      - strip leading/trailing whitespace per line
      - remove null bytes
    """
    text = text.replace("\x00", "")
    lines = [line.strip() for line in text.splitlines()]

    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    return "\n".join(cleaned).strip()
