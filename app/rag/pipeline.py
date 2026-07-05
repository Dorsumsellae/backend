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


def _where(workspace: str, filename: str | None = None) -> dict:
    """Construit le filtre ChromaDB restreignant a un workspace (et un document).

    ChromaDB exige un operateur `$and` explicite pour combiner plusieurs cles ;
    avec une seule condition, on renvoie le filtre simple attendu.
    """
    clauses: list[dict] = [{"workspace": workspace}]
    if filename:
        clauses.append({"filename": filename})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


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


def _store_passages(workspace: str, filename: str, passages: list[dict]) -> int:
    """Remplace les passages d'un document dans le vectorstore (upsert idempotent).

    Chaque element de `passages` est un dict `{"text", "metadata"}` ; la metadonnee
    doit contenir au moins `passage_id`. Les ids sont prefixes par le workspace,
    si bien qu'une reindexation du meme fichier ecrase ses propres lignes et que
    deux workspaces peuvent heberger un fichier homonyme sans collision.
    """
    if not passages:
        return 0

    texts = [passage["text"] for passage in passages]
    metadatas = [passage["metadata"] for passage in passages]
    ids = [f"{workspace}:{filename}:{meta['passage_id']}" for meta in metadatas]

    store = get_vectorstore()
    # Purge d'une eventuelle indexation precedente du meme document : evite les
    # passages orphelins si le document a ete raccourci depuis.
    store._collection.delete(where=_where(workspace, filename))
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(passages)


def index_document(
    filename: str, text: str, workspace: str, strategy: str | None = None
) -> int:
    """Decoupe le document, l'ajoute au vectorstore et retourne le nombre de passages.

    L'indexation est idempotente : reindexer un meme fichier (dans le meme
    `workspace`) remplace ses passages au lieu d'en accumuler des doublons.

    `strategy` selectionne le decoupage ("fixed" ou "recursive") ; par defaut,
    la valeur de configuration `settings.chunk_strategy` est utilisee.
    """
    chunks = split_text(text, strategy=strategy)
    passages = [
        {
            "text": chunk.text,
            "metadata": {
                "workspace": workspace,
                "filename": filename,
                "passage_id": chunk.passage_id,
                "excerpt": _excerpt(chunk.text),
            },
        }
        for chunk in chunks
    ]
    return _store_passages(workspace, filename, passages)


def index_transcript(
    filename: str, cues: list, workspace: str, source_url: str | None = None
) -> int:
    """Indexe un transcript horodate : decoupe temporel + metadonnees d'instant.

    Chaque passage conserve `start`/`end` (secondes) ; si `source_url` est fourni
    (ex. URL YouTube), il est stocke pour permettre des sources cliquables pointant
    vers l'instant exact de la video. Idempotent comme `index_document`.
    """
    from app.rag.transcripts import chunk_cues

    chunks = chunk_cues(cues, settings.chunk_size, settings.chunk_overlap)
    passages = []
    for chunk in chunks:
        metadata = {
            "workspace": workspace,
            "filename": filename,
            "passage_id": chunk.passage_id,
            "excerpt": _excerpt(chunk.text),
            "content_type": "transcript",
            "start": round(chunk.start, 3),
            "end": round(chunk.end, 3),
        }
        if source_url:
            metadata["source_url"] = source_url
        if chunk.speaker:
            metadata["speaker"] = chunk.speaker
        passages.append({"text": chunk.text, "metadata": metadata})
    return _store_passages(workspace, filename, passages)


def list_indexed_documents(workspace: str) -> list[dict]:
    """Retourne les documents indexes d'un `workspace`, avec leur nombre de passages.

    Les passages sont regroupes par `filename` (metadonnee posee a l'indexation),
    ce qui donne un document logique par fichier, trie par nom.
    """
    store = get_vectorstore()
    data = store._collection.get(where=_where(workspace), include=["metadatas"])

    counts: dict[str, int] = {}
    for metadata in data.get("metadatas") or []:
        filename = (metadata or {}).get("filename")
        if filename:
            counts[filename] = counts.get(filename, 0) + 1

    return [
        {"filename": filename, "chunks_indexed": chunks}
        for filename, chunks in sorted(counts.items())
    ]


def list_workspaces() -> list[str]:
    """Retourne la liste triee des workspaces contenant au moins un passage indexe."""
    store = get_vectorstore()
    data = store._collection.get(include=["metadatas"])
    workspaces = {
        (metadata or {}).get("workspace")
        for metadata in (data.get("metadatas") or [])
        if (metadata or {}).get("workspace")
    }
    return sorted(workspaces)


def reset_index(workspace: str, filename: str | None = None) -> dict:
    """Reinitialise l'indexation d'un `workspace` et retourne un compte-rendu.

    - `filename` fourni : desindexe uniquement ce document du workspace.
    - `filename` omis : vide tous les documents du workspace (les autres
      workspaces restent intacts).

    Le compte-rendu (`documents_removed`, `chunks_removed`) est calcule avant
    la suppression pour renvoyer au front ce qui a reellement ete retire.
    """
    store = get_vectorstore()
    where = _where(workspace, filename)
    data = store._collection.get(where=where, include=["metadatas"])
    chunks_removed = len(data.get("ids") or [])
    filenames = {
        (metadata or {}).get("filename")
        for metadata in (data.get("metadatas") or [])
        if (metadata or {}).get("filename")
    }
    if chunks_removed:
        store._collection.delete(where=where)

    return {
        "scope": "document" if filename else "workspace",
        "documents_removed": (1 if chunks_removed else 0) if filename else len(filenames),
        "chunks_removed": chunks_removed,
    }


def answer_question(
    question: str,
    workspace: str,
    top_k: int | None = None,
    model: str | None = None,
    filename: str | None = None,
) -> dict:
    """Recherche les passages proches, interroge le LLM et retourne reponse + sources.

    La recherche est **restreinte au `workspace`** (et, si `filename` est fourni,
    a ce seul document) : jamais de fuite entre espaces de travail.

    `model` selectionne le modele Ollama repondant (defaut : settings.ollama_model).
    Le modele reellement utilise est renvoye dans la cle "model".
    """
    k = top_k or settings.top_k
    model_name = model or settings.ollama_model
    results = get_vectorstore().similarity_search_with_score(
        question, k=k, filter=_where(workspace, filename)
    )

    passages = [document.page_content for document, _ in results]
    sources = [_build_source(document, score) for document, score in results]

    prompt = build_prompt(question, passages)
    answer = get_llm(model_name).invoke(prompt)
    return {"answer": answer, "sources": sources, "model": model_name}


def _build_source(document, score) -> dict:
    """Construit une source d'affichage, enrichie de l'instant video si transcript."""
    metadata = document.metadata
    source = {
        "filename": metadata.get("filename", ""),
        "passage_id": metadata.get("passage_id", -1),
        "excerpt": metadata.get("excerpt", ""),
        "score": float(score),
    }
    start = metadata.get("start")
    if start is not None:
        source["start_seconds"] = float(start)
    source_url = metadata.get("source_url")
    if source_url:
        source["source_url"] = source_url
        source["timecode_url"] = _timecode_url(source_url, start)
    speaker = metadata.get("speaker")
    if speaker:
        source["speaker"] = speaker
    return source


def _timecode_url(url: str, start: float | None) -> str:
    """Ajoute un ancrage temporel `t=<secondes>s` a une URL video (ex. YouTube)."""
    if start is None:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={int(start)}s"


def _excerpt(text: str) -> str:
    """Tronque un passage pour ne stocker qu'un court extrait lisible."""
    if len(text) <= EXCERPT_MAX_CHARS:
        return text
    return text[:EXCERPT_MAX_CHARS].rstrip() + "..."
