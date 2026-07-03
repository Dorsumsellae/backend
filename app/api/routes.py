"""Endpoints de l'API RAG.

    GET  /health   -> disponibilite du service
    POST /upload   -> envoi du document (stockage MinIO)
    POST /index    -> indexation du document (chunks -> embeddings -> ChromaDB)
    POST /ask      -> question -> reponse generee + sources
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import (
    AskRequest,
    AskResponse,
    IndexRequest,
    IndexResponse,
)
from app.rag import pipeline
from app.storage import minio_client

router = APIRouter()


@router.get("/health", tags=["health"])
def health() -> dict:
    """Endpoint de sante (utilise par la CI et le healthcheck Docker)."""
    return {"status": "ok"}


@router.post("/upload", tags=["rag"])
async def upload(file: UploadFile = File(...)) -> dict:
    """Recoit un document et le stocke dans MinIO."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")
    content = await file.read()
    filename = minio_client.put_document(file.filename, content)
    return {"filename": filename}


@router.post("/index", response_model=IndexResponse, tags=["rag"])
def index(req: IndexRequest) -> IndexResponse:
    """Indexe un document deja stocke dans MinIO."""
    text = minio_client.get_document(req.filename)
    chunks_indexed = pipeline.index_document(req.filename, text)
    return IndexResponse(filename=req.filename, chunks_indexed=chunks_indexed)


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Repond a une question a partir du document indexe."""
    result = pipeline.answer_question(req.question, top_k=req.top_k)
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
    )
