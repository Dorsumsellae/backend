"""Chaine RAG de bout en bout : indexation et interrogation.

Ce module assemble les briques (chunking, embeddings, vectorstore, prompt, LLM).
Les fonctions sont volontairement laissees a completer (coeur du TP).
"""

from langchain_ollama import OllamaLLM

from app.config import settings
from app.rag.chunking import split_text
from app.rag.prompt import build_prompt
from app.rag.vectorstore import get_vectorstore


def get_llm() -> OllamaLLM:
    """Retourne le client LLM Ollama configure."""
    return OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url)


def index_document(filename: str, text: str) -> int:
    """Indexe un document dans la base vectorielle.

    Etapes :
        1. decouper `text` en passages (split_text) ;
        2. construire les metadonnees (filename, passage_id, extrait) ;
        3. ajouter les passages au vectorstore (embeddings automatiques).

    Returns:
        Le nombre de passages indexes.

    TODO (etudiant) : implementer l'ajout au vectorstore.
    """
    chunks = split_text(text)
    _ = chunks  # a stocker via get_vectorstore().add_texts(...)
    raise NotImplementedError


def answer_question(question: str, top_k: int | None = None) -> dict:
    """Repond a une question a partir des passages indexes.

    Etapes :
        1. recherche de similarite (top_k passages) ;
        2. construction du prompt enrichi (build_prompt) ;
        3. generation via le LLM ;
        4. retour de la reponse + sources.

    TODO (etudiant) : implementer la recherche et l'appel LLM.
    """
    k = top_k or settings.top_k
    _store = get_vectorstore()
    _ = (k, _store, build_prompt)
    raise NotImplementedError
