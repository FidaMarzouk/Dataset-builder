import requests
from .base import AppClient

RAG_API_URL = "http://localhost:8000/run_for_eval"

def rag_pgvector(question: str, required_fields: set[str] | None = None) -> dict:
    resp = requests.post(
        RAG_API_URL,
        json={"question": question, "required_fields": list(required_fields) if required_fields else None},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()