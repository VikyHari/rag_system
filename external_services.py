"""
External Service Layer (Open-Source)
====================================
Uses Ollama running Qwen models locally. No paid API keys needed.
Falls back to Gemini if configured, but primary path is fully local.
"""

from typing import Dict, Optional

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GEMINI_API_KEY,
    DEFAULT_LLM,
)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly on the "
    "provided context. If the context does not contain enough information to "
    "answer the question, say so clearly. Do not make up facts."
)


def _build_prompt(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based on the context above:"
    )


def ask_qwen(context: str, question: str, model: Optional[str] = None) -> str:
    """Generate an answer using Ollama running a Qwen model locally."""
    model = model or OLLAMA_MODEL
    prompt = _build_prompt(context, question)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 1024,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def ask_ollama(context: str, question: str, model: Optional[str] = None) -> str:
    """Generic Ollama call — same as ask_qwen but lets you specify any model."""
    return ask_qwen(context, question, model=model)


def ask_gemini(context: str, question: str, model: str = "gemini-1.5-flash") -> str:
    """Generate an answer using Google Gemini (optional paid fallback)."""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Please use Qwen (local) instead."

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"{SYSTEM_PROMPT}\n\n{_build_prompt(context, question)}"

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


LLM_PROVIDERS = {
    "qwen": ask_qwen,
    "ollama": ask_ollama,
    "gemini": ask_gemini,
}


def generate_answer(
    context: str,
    question: str,
    llm_name: Optional[str] = None,
) -> Dict[str, str]:
    """Route a question to the appropriate LLM."""
    provider = (llm_name or DEFAULT_LLM).lower()
    if provider not in LLM_PROVIDERS:
        provider = "qwen"

    fn = LLM_PROVIDERS[provider]
    try:
        answer = fn(context, question)
    except requests.ConnectionError:
        return {
            "answer": (
                "Could not connect to Ollama. Please make sure Ollama is running:\n"
                "1. Install from https://ollama.com\n"
                "2. Run: ollama pull qwen2.5:7b\n"
                "3. Ollama starts automatically after install"
            ),
            "llm_used": provider,
        }
    except Exception as e:
        return {"answer": f"Error from {provider}: {str(e)}", "llm_used": provider}

    return {"answer": answer, "llm_used": provider}
