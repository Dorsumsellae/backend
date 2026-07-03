"""Endpoints de l'API RAG.

    GET  /health    -> disponibilite du service
    POST /upload    -> envoi du document (stockage MinIO)
    POST /index     -> indexation du document (chunks -> embeddings -> ChromaDB)
    GET  /documents -> liste des documents indexes
    POST /ask       -> question -> reponse generee + sources
    POST /reset     -> reinitialise l'indexation (tout ou un document)
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from minio.error import S3Error

from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentsResponse,
    IndexRequest,
    IndexResponse,
    ResetRequest,
    ResetResponse,
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
    try:
        text = minio_client.get_document(req.filename)
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document '{req.filename}' introuvable dans MinIO. "
                    "Uploadez-le d'abord via /upload."
                ),
            ) from exc
        raise
    chunks_indexed = pipeline.index_document(req.filename, text, strategy=req.strategy)
    return IndexResponse(filename=req.filename, chunks_indexed=chunks_indexed)


@router.get("/documents", response_model=DocumentsResponse, tags=["rag"])
def documents() -> DocumentsResponse:
    """Liste les documents actuellement indexes dans ChromaDB."""
    docs = pipeline.list_indexed_documents()
    return DocumentsResponse(
        documents=[DocumentInfo(**doc) for doc in docs],
        count=len(docs),
    )


@router.post("/reset", response_model=ResetResponse, tags=["rag"])
def reset(req: ResetRequest | None = None) -> ResetResponse:
    """Reinitialise l'indexation.

    Corps optionnel : sans corps (ou `filename` a null) toute la collection est
    videe ; avec `{"filename": "..."}` seul ce document est desindexe.
    """
    filename = req.filename if req else None
    result = pipeline.reset_index(filename)
    return ResetResponse(**result)


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Repond a une question a partir du document indexe."""
    result = pipeline.answer_question(req.question, top_k=req.top_k)
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
    )
