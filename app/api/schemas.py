"""Schemas Pydantic pour les requetes et reponses de l'API."""

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    filename: str = Field(..., description="Nom du fichier a indexer (dans MinIO).")


class IndexResponse(BaseModel):
    filename: str
    chunks_indexed: int


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
