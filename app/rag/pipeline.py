"""Chaine RAG de bout en bout : indexation et interrogation.

Ce module assemble les briques (chunking, embeddings, vectorstore, prompt, LLM).
Les fonctions sont volontairement laissees a completer (coeur du TP).
"""

from langchain_ollama import OllamaLLM

from app.config import settings
from app.rag.chunking import split_text
from app.rag.prompt import build_prompt
from app.rag.vectorstore import get_vectorstore, reset_collection

# Longueur de l'extrait conserve dans les metadonnees pour l'affichage des sources.
EXCERPT_MAX_CHARS = 200


def get_llm(model: str | None = None) -> OllamaLLM:
    """Retourne le client LLM Ollama pour `model` (defaut : settings.ollama_model)."""
    return OllamaLLM(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
    )


def list_available_models() -> list[str]:
    """Liste les modeles disponibles cote Ollama, tries par nom.

    Interroge l'API `GET {ollama_base_url}/api/tags`. Permet au front de
    proposer un selecteur de modele. Leve `requests.RequestException` si le
    serveur Ollama est injoignable (a traiter par l'appelant).
    """
    import requests

    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    models = response.json().get("models") or []
    return sorted(m["name"] for m in models if m.get("name"))


def index_document(filename: str, text: str, strategy: str | None = None) -> int:
    """Decoupe le document, l'ajoute au vectorstore et retourne le nombre de passages.

    L'indexation est idempotente : reindexer un meme fichier remplace ses
    passages au lieu d'en accumuler des doublons.

    `strategy` selectionne le decoupage ("fixed" ou "recursive") ; par defaut,
    la valeur de configuration `settings.chunk_strategy` est utilisee.
    """
    chunks = split_text(text, strategy=strategy)
    if not chunks:
        return 0

    texts = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "filename": filename,
            "passage_id": chunk.passage_id,
            "excerpt": _excerpt(chunk.text),
        }
        for chunk in chunks
    ]
    # Identifiants deterministes : une reindexation du meme fichier ecrase les
    # memes lignes (upsert) au lieu de creer des doublons.
    ids = [f"{filename}:{chunk.passage_id}" for chunk in chunks]

    store = get_vectorstore()
    # Purge d'une eventuelle indexation precedente du meme document : evite les
    # passages orphelins si le document a ete raccourci depuis.
    store._collection.delete(where={"filename": filename})
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(chunks)


def list_indexed_documents() -> list[dict]:
    """Retourne les documents indexes dans ChromaDB, avec leur nombre de passages.

    Les passages sont regroupes par `filename` (metadonnee posee a l'indexation),
    ce qui donne un document logique par fichier, trie par nom.
    """
    store = get_vectorstore()
    data = store._collection.get(include=["metadatas"])

    counts: dict[str, int] = {}
    for metadata in data.get("metadatas") or []:
        filename = (metadata or {}).get("filename")
        if filename:
            counts[filename] = counts.get(filename, 0) + 1

    return [
        {"filename": filename, "chunks_indexed": chunks}
        for filename, chunks in sorted(counts.items())
    ]


def reset_index(filename: str | None = None) -> dict:
    """Reinitialise l'indexation et retourne un compte-rendu de la suppression.

    - `filename` fourni : desindexe uniquement ce document (les autres restent).
    - `filename` omis : vide entierement la collection ChromaDB.

    Le compte-rendu (`documents_removed`, `chunks_removed`) est calcule avant
    la suppression pour renvoyer au front ce qui a reellement ete retire.
    """
    store = get_vectorstore()

    if filename:
        existing = store._collection.get(where={"filename": filename}, include=[])
        chunks_removed = len(existing.get("ids") or [])
        if chunks_removed:
            store._collection.delete(where={"filename": filename})
        return {
            "scope": "document",
            "documents_removed": 1 if chunks_removed else 0,
            "chunks_removed": chunks_removed,
        }

    # Reset complet : on inventorie avant de vider pour renvoyer les totaux.
    data = store._collection.get(include=["metadatas"])
    filenames = {
        (metadata or {}).get("filename")
        for metadata in (data.get("metadatas") or [])
        if (metadata or {}).get("filename")
    }
    chunks_removed = len(data.get("ids") or [])
    reset_collection()
    return {
        "scope": "all",
        "documents_removed": len(filenames),
        "chunks_removed": chunks_removed,
    }


def answer_question(
    question: str, top_k: int | None = None, model: str | None = None
) -> dict:
    """Recherche les passages proches, interroge le LLM et retourne reponse + sources.

    `model` selectionne le modele Ollama repondant (defaut : settings.ollama_model).
    Le modele reellement utilise est renvoye dans la cle "model".
    """
    k = top_k or settings.top_k
    model_name = model or settings.ollama_model
    results = get_vectorstore().similarity_search_with_score(question, k=k)

    passages = [document.page_content for document, _ in results]
    sources = [
        {
            "filename": document.metadata.get("filename", ""),
            "passage_id": document.metadata.get("passage_id", -1),
            "excerpt": document.metadata.get("excerpt", ""),
            "score": float(score),
        }
        for document, score in results
    ]

    prompt = build_prompt(question, passages)
    answer = get_llm(model_name).invoke(prompt)
    return {"answer": answer, "sources": sources, "model": model_name}


def _excerpt(text: str) -> str:
    """Tronque un passage pour ne stocker qu'un court extrait lisible."""
    if len(text) <= EXCERPT_MAX_CHARS:
        return text
    return text[:EXCERPT_MAX_CHARS].rstrip() + "..."
