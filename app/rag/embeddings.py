"""Modele d'embeddings multilingue (sentence-transformers via LangChain).

Le modele est charge une seule fois (singleton) car son initialisation est
couteuse. Le telechargement depuis HuggingFace a lieu au premier appel.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Retourne l'instance partagee du modele d'embeddings."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
