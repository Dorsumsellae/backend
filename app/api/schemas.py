"""Schemas Pydantic pour les requetes et reponses de l'API."""

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    filename: str = Field(..., description="Nom du fichier a indexer (dans MinIO).")


class IndexResponse(BaseModel):
    filename: str
    chunks_indexed: int


class ResetRequest(BaseModel):
    filename: str | None = Field(
        None,
        description=(
            "Document a desindexer. Si omis (ou null), toute l'indexation "
            "est reinitialisee."
        ),
    )


class ResetResponse(BaseModel):
    scope: str = Field(
        ..., description="'all' (toute la collection) ou 'document' (un seul fichier)."
    )
    documents_removed: int = Field(
        ..., description="Nombre de documents distincts desindexes."
    )
    chunks_removed: int = Field(..., description="Nombre de passages supprimes.")


class DocumentInfo(BaseModel):
    filename: str = Field(..., description="Nom du document indexe.")
    chunks_indexed: int = Field(
        ..., description="Nombre de passages (chunks) indexes pour ce document."
    )


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    count: int = Field(..., description="Nombre de documents indexes distincts.")


class AskRequest(BaseModel):
    question: str = Field(..., description="Question en langage naturel.")
    top_k: int | None = Field(None, description="Nombre de passages a recuperer.")


class Source(BaseModel):
    filename: str
    passage_id: int
    excerpt: str
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
