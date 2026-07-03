"""Chaine RAG de bout en bout : indexation et interrogation.

Ce module assemble les briques (chunking, embeddings, vectorstore, prompt, LLM).
Les fonctions sont volontairement laissees a completer (coeur du TP).
"""

from langchain_ollama import OllamaLLM

from app.config import settings
from app.rag.chunking import split_text
from app.rag.prompt import build_prompt
from app.rag.vectorstore import get_vectorstore

# Longueur de l'extrait conserve dans les metadonnees pour l'affichage des sources.
EXCERPT_MAX_CHARS = 200


def get_llm() -> OllamaLLM:
    """Retourne le client LLM Ollama configure."""
    return OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url)


def index_document(filename: str, text: str) -> int:
    """Decoupe le document, l'ajoute au vectorstore et retourne le nombre de passages.

    L'indexation est idempotente : reindexer un meme fichier remplace ses
    passages au lieu d'en accumuler des doublons.
    """
    chunks = split_text(text)
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


def answer_question(question: str, top_k: int | None = None) -> dict:
    """Recherche les passages proches, interroge le LLM et retourne reponse + sources."""
    k = top_k or settings.top_k
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
    answer = get_llm().invoke(prompt)
    return {"answer": answer, "sources": sources}


def _excerpt(text: str) -> str:
    """Tronque un passage pour ne stocker qu'un court extrait lisible."""
    if len(text) <= EXCERPT_MAX_CHARS:
        return text
    return text[:EXCERPT_MAX_CHARS].rstrip() + "..."
