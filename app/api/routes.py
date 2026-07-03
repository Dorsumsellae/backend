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

router = APIRouter()


@router.get("/health", tags=["health"])
def health() -> dict:
    """Endpoint de sante (utilise par la CI et le healthcheck Docker)."""
    return {"status": "ok"}


@router.post("/upload", tags=["rag"])
async def upload(file: UploadFile = File(...)) -> dict:
    """Recoit un document et le stocke dans MinIO.

    TODO (etudiant) :
        - lire le contenu du fichier ;
        - le pousser dans le bucket MinIO via app.storage.minio_client ;
        - retourner le nom du fichier stocke.
    """
    raise HTTPException(status_code=501, detail="Non implemente : voir upload().")


@router.post("/index", response_model=IndexResponse, tags=["rag"])
def index(req: IndexRequest) -> IndexResponse:
    """Indexe un document deja stocke.

    TODO (etudiant) :
        - recuperer le texte depuis MinIO ;
        - decouper en chunks (app.rag.chunking) ;
        - creer les embeddings et stocker dans ChromaDB (app.rag.vectorstore) ;
        - retourner le nombre de passages indexes.
    """
    raise HTTPException(status_code=501, detail="Non implemente : voir index().")


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Repond a une question a partir du document indexe.

    TODO (etudiant) :
        - transformer la question en embedding ;
        - rechercher les top_k passages proches dans ChromaDB ;
        - construire le prompt enrichi (app.rag.prompt) ;
        - interroger le LLM Ollama et retourner reponse + sources.
    """
    raise HTTPException(status_code=501, detail="Non implemente : voir ask().")
