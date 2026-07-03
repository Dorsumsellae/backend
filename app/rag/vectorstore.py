"""Acces a la base vectorielle ChromaDB (via LangChain)."""

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma

from app.config import settings
from app.rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Retourne le store ChromaDB connecte au serveur (mode client/serveur)."""
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
    )


def reset_collection() -> None:
    """Vide la collection vectorielle (bonus : bouton de reinitialisation).

    TODO (etudiant) : supprimer puis recreer la collection.
    """
    raise NotImplementedError
