"""Endpoints de l'API RAG.

    GET  /health    -> disponibilite du service
    POST /upload    -> envoi du document (stockage MinIO)
    POST /index     -> indexation du document (chunks -> embeddings -> ChromaDB)
    GET  /documents -> liste des documents indexes
    GET  /models    -> liste des modeles Ollama disponibles
    POST /ask       -> question -> reponse generee + sources
    POST /reset     -> reinitialise l'indexation (tout ou un document)
"""

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from minio.error import S3Error

from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentsResponse,
    IndexRequest,
    IndexResponse,
    ModelInfo,
    ModelsResponse,
    ResetRequest,
    ResetResponse,
)
from app.config import settings
from app.rag import loaders, pipeline
from app.storage import minio_client

router = APIRouter()


@router.get("/health", tags=["health"])
def health() -> dict:
    """Endpoint de sante (utilise par la CI et le healthcheck Docker)."""
    return {"status": "ok"}


@router.post("/upload", tags=["rag"])
async def upload(file: UploadFile = File(...)) -> dict:
    """Recoit un document (texte ou PDF non scanne) et le stocke dans MinIO."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")
    if not loaders.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format non pris en charge. Formats acceptes : "
                f"{', '.join(sorted(loaders.SUPPORTED_EXTENSIONS))}."
            ),
        )
    content = await file.read()
    filename = minio_client.put_document(
        file.filename, content, content_type=file.content_type
    )
    return {"filename": filename}


@router.post("/index", response_model=IndexResponse, tags=["rag"])
def index(req: IndexRequest) -> IndexResponse:
    """Indexe un document deja stocke dans MinIO (texte extrait selon le format)."""
    try:
        content = minio_client.get_document_bytes(req.filename)
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

    try:
        text = loaders.extract_text(req.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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


@router.get("/models", response_model=ModelsResponse, tags=["rag"])
def models() -> ModelsResponse:
    """Liste les modeles Ollama disponibles (pour alimenter le selecteur du front)."""
    try:
        names = pipeline.list_available_models()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Serveur Ollama injoignable : {exc}",
        ) from exc
    return ModelsResponse(
        models=[
            ModelInfo(name=name, is_default=(name == settings.ollama_model))
            for name in names
        ],
        default=settings.ollama_model,
    )


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Repond a une question a partir du document indexe."""
    result = pipeline.answer_question(req.question, top_k=req.top_k, model=req.model)
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
    )
