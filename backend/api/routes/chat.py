
from fastapi import APIRouter
from fastapi import APIRouter
from backend.rag.rag_pipeline2 import rag_generate_ollama, retriever_global

router = APIRouter()

@router.post("/", response_model=str)
def chat_endpoint(query: str):
    return rag_generate_ollama(query, retriever_global)
