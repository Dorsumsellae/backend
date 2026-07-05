"""Chaine RAG de bout en bout : indexation et interrogation.

Ce module assemble les briques (chunking, embeddings, vectorstore, prompt, LLM).
Les fonctions sont volontairement laissees a completer (coeur du TP).
"""

from langchain_ollama import OllamaLLM

from app.config import settings
from app.rag import retrieval
from app.rag.chunking import split_text
from app.rag.prompt import (
    build_chat_prompt,
    build_prompt,
    build_router_prompt,
    build_summary_prompt,
    cited_indices,
)
from app.rag.vectorstore import get_vectorstore

# Longueur de l'extrait conserve dans les metadonnees pour l'affichage des sources.
EXCERPT_MAX_CHARS = 200


def _where(
    workspace: str,
    filename: str | None = None,
    filenames: list[str] | None = None,
) -> dict:
    """Construit le filtre ChromaDB restreignant a un workspace (et des documents).

    Trois portees possibles :
      - tout le workspace (ni `filename` ni `filenames`) ;
      - un seul document (`filename`) ;
      - un sous-ensemble de documents (`filenames`, via l'operateur `$in`).

    `filename` (document unique) est prioritaire si les deux sont fournis. Une liste
    `filenames` d'un seul element retombe sur une egalite simple, et une liste vide
    est ignoree (= tout le workspace) : on n'emet jamais `{"$in": []}` qui ne
    matcherait rien. ChromaDB exige un `$and` explicite pour combiner plusieurs cles.
    """
    clauses: list[dict] = [{"workspace": workspace}]
    if filename:
        clauses.append({"filename": filename})
    elif filenames:
        clauses.append(
            {"filename": filenames[0]}
            if len(filenames) == 1
            else {"filename": {"$in": filenames}}
        )
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def get_llm(
    model: str | None = None,
    num_ctx: int | None = None,
    temperature: float | None = None,
) -> OllamaLLM:
    """Retourne le client LLM Ollama pour `model` (defaut : settings.ollama_model).

    `num_ctx` fixe la fenetre de contexte cote Ollama (defaut : settings.ollama_num_ctx).
    Sans ce reglage, Ollama plafonne a ~4096 tokens quelle que soit la capacite du
    modele, ce qui tronquerait un prompt a fort top_k ou une synthese globale.

    `temperature` surcharge la temperature de generation (defaut :
    settings.ollama_temperature). On teste `is None` explicitement pour autoriser 0.0
    (generation deterministe, ex. pour le routeur d'intention).
    """
    return OllamaLLM(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        num_ctx=num_ctx or settings.ollama_num_ctx,
        temperature=(
            settings.ollama_temperature if temperature is None else temperature
        ),
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


def _document_type(filename: str, metadata: dict) -> str:
    """Devine le type/origine d'une source a partir de ses metadonnees de passage.

    Renvoie 'youtube' (transcript video avec URL), 'transcript' (sous-titres sans
    URL), 'pdf', ou 'text'. Sert a l'affichage (icone de source cote front).
    """
    is_transcript = metadata.get("content_type") == "transcript" or filename.startswith("youtube_")
    if is_transcript:
        if metadata.get("source_url") or filename.startswith("youtube_"):
            return "youtube"
        return "transcript"
    if filename.lower().endswith(".pdf"):
        return "pdf"
    return "text"


def list_indexed_documents(workspace: str) -> list[dict]:
    """Retourne les documents indexes d'un `workspace`, avec leur nombre de passages.

    Les passages sont regroupes par `filename` (metadonnee posee a l'indexation),
    ce qui donne un document logique par fichier, trie par nom. Chaque entree porte
    aussi un `type` (text/pdf/youtube/transcript) et un `source_url` eventuel,
    derives des metadonnees deja presentes (aucun read supplementaire).
    """
    store = get_vectorstore()
    data = store._collection.get(where=_where(workspace), include=["metadatas"])

    counts: dict[str, int] = {}
    types: dict[str, str] = {}
    urls: dict[str, str | None] = {}
    for metadata in data.get("metadatas") or []:
        metadata = metadata or {}
        filename = metadata.get("filename")
        if not filename:
            continue
        counts[filename] = counts.get(filename, 0) + 1
        types.setdefault(filename, _document_type(filename, metadata))
        if not urls.get(filename) and metadata.get("source_url"):
            urls[filename] = metadata.get("source_url")

    return [
        {
            "filename": filename,
            "chunks_indexed": counts[filename],
            "type": types.get(filename, "text"),
            "source_url": urls.get(filename),
        }
        for filename in sorted(counts)
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
        "workspace": workspace,
        "documents_removed": (1 if chunks_removed else 0) if filename else len(filenames),
        "chunks_removed": chunks_removed,
    }


def _search_sources(question: str, workspace: str, k: int, filename=None, filenames=None):
    """Recherche les passages pertinents et renvoie (passages_texte, sources).

    Le retrieval avance (dense/MMR + BM25/RRF -> reranking -> reorder) est delegue
    a `app.rag.retrieval`. Chaque source recoit un index de citation `cite` (1-based)
    egal a sa position dans le contexte du prompt, ce qui permet au front de lier
    reponse et sources.
    """
    store = get_vectorstore()
    where = _where(workspace, filename, filenames)
    scored = retrieval.retrieve(store, question, k, where)
    passages = [document.page_content for document, _ in scored]
    sources = [
        {**_build_source(document, score), "cite": i + 1}
        for i, (document, score) in enumerate(scored)
    ]
    return passages, sources


# --- Mode resume : routage d'intention + synthese globale --------------------

# Mots-cles declenchant SANS ambiguite une synthese globale (route directe, sans
# appel LLM). Les autres cas sont tranches par le routeur LLM (cf. _route_is_summary).
_SUMMARY_KEYWORDS = (
    "resume",  # resume / resumer / resumé (sous-chaine commune)
    "de quoi parle",
    "de quoi ca parle",
    "de quoi ça parle",
    "parle de quoi",
    "sujet principal",
    "sujet de la",
    "sujet du",
    "themes",
    "thèmes",
    "synthese",
    "synthèse",
    "vue d'ensemble",
    "apercu",
    "aperçu",
    "grandes lignes",
    "idee generale",
    "idée générale",
    "de quoi il s'agit",
    "de quoi s'agit",
)


def _keyword_is_summary(question: str) -> bool:
    """Heuristique : la question demande-t-elle clairement une synthese globale ?"""
    q = (question or "").lower()
    return any(keyword in q for keyword in _SUMMARY_KEYWORDS)


def _llm_route_is_summary(question: str, model: str) -> bool:
    """Routeur LLM : demande au modele de classer l'intention (resume vs factuel).

    Repli sur False (reponse factuelle = RAG classique) en cas de reponse inattendue
    ou d'erreur : le comportement par defaut reste le retrieval top-k.
    """
    try:
        # Temperature 0 : classification stable (RESUME/FACTUEL), pas de creativite.
        verdict = get_llm(model, temperature=0.0).invoke(build_router_prompt(question))
    except Exception:
        return False
    low = (verdict or "").strip().lower()
    return "resum" in low or "résum" in low


def _route_is_summary(question: str, model: str) -> bool:
    """Routage hybride : mots-cles evidents d'abord, sinon routeur LLM (Gemma decide)."""
    if _keyword_is_summary(question):
        return True
    return _llm_route_is_summary(question, model)


def _order_index(metadata: dict, fallback: int) -> tuple[int, float]:
    """Cle de tri chronologique d'un chunk : instant `start` sinon `passage_id`."""
    metadata = metadata or {}
    start = metadata.get("start")
    if start is not None:
        return (0, float(start))
    passage_id = metadata.get("passage_id")
    if passage_id is not None:
        return (0, float(passage_id))
    return (1, float(fallback))


def _sample_passages(store, where: dict, n: int) -> tuple[list[str], list[dict]]:
    """Echantillonne jusqu'a `n` passages repartis sur TOUT le perimetre.

    Recupere les chunks du perimetre, les ordonne (instant `start` pour un transcript,
    sinon `passage_id`), puis retient `n` passages equidistants du debut a la fin.
    Convient a une synthese globale (couverture du document > pertinence locale).
    """
    data = store._collection.get(where=where, include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        return [], []

    order = sorted(
        range(len(documents)),
        key=lambda i: _order_index(metadatas[i], i),
    )
    if len(order) <= n:
        chosen = order
    else:
        step = len(order) / n
        chosen = [order[int(rank * step)] for rank in range(n)]

    passages = [documents[i] for i in chosen]
    metas = [metadatas[i] or {} for i in chosen]
    return passages, metas


def summarize(
    workspace: str,
    question: str | None = None,
    model: str | None = None,
    filename: str | None = None,
    filenames: list[str] | None = None,
) -> dict:
    """Synthetise le document (ou le perimetre) a partir d'un echantillon large.

    Repond aux questions globales (« de quoi parle... », « resume ») pour lesquelles le
    retrieval top-k est inadapte : on echantillonne des passages couvrant tout le
    document (cf. `_sample_passages`) et on demande une synthese dediee (sans citation
    ni phrase de refus). Meme forme de retour que `answer_question`.
    """
    from langchain_core.documents import Document

    model_name = model or settings.ollama_model
    store = get_vectorstore()
    where = _where(workspace, filename, filenames)
    passages, metas = _sample_passages(store, where, settings.summary_sample_size)

    if not passages:
        return {
            "answer": "Je ne trouve pas cette information dans le document fourni.",
            "sources": [],
            "model": model_name,
            "cited": [],
        }

    answer = get_llm(model_name).invoke(build_summary_prompt(passages, question))
    docs = [Document(page_content=p, metadata=m) for p, m in zip(passages, metas)]
    sources = [
        {**_build_source(document, None), "cite": i + 1}
        for i, document in enumerate(docs)
    ]
    return {
        "answer": answer,
        "sources": sources,
        "model": model_name,
        "cited": [],
    }


def answer_question(
    question: str,
    workspace: str,
    top_k: int | None = None,
    model: str | None = None,
    filename: str | None = None,
    filenames: list[str] | None = None,
) -> dict:
    """Recherche les passages proches, interroge le LLM et retourne reponse + sources.

    La recherche est **restreinte au `workspace`**, et facultativement a un seul
    document (`filename`) ou a un sous-ensemble (`filenames`) : jamais de fuite
    entre espaces de travail.

    Les questions **globales** (« de quoi parle... », « resume ») sont routees vers
    `summarize` (synthese sur un echantillon couvrant tout le document) ; les autres
    suivent le RAG top-k classique.

    `model` selectionne le modele Ollama repondant (defaut : settings.ollama_model).
    Le modele reellement utilise est renvoye dans la cle "model" ; "cited" liste les
    numeros de passages `[n]` effectivement cites par le modele.
    """
    k = top_k or settings.top_k
    model_name = model or settings.ollama_model

    if _route_is_summary(question, model_name):
        return summarize(workspace, question, model_name, filename, filenames)

    passages, sources = _search_sources(question, workspace, k, filename, filenames)

    prompt = build_prompt(question, passages)
    answer = get_llm(model_name).invoke(prompt)
    return {
        "answer": answer,
        "sources": sources,
        "model": model_name,
        "cited": cited_indices(answer, len(sources)),
    }


def _latest_user_message(messages: list[dict]) -> str:
    """Retourne le contenu du dernier message de role 'user'."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def _retrieval_query(messages: list[dict]) -> str:
    """Construit la requete de recherche d'un tour de chat.

    On concatene les dernieres questions **utilisateur** (jusqu'a 3). La question
    courante peut etre elliptique (« et qui l'a lance ? ») et ne plus contenir le
    sujet : y adjoindre les questions precedentes reinjecte les termes de contexte
    (coreference) et fiabilise la recherche des passages sur les follow-ups.
    """
    user_turns = [
        (m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "user"
    ]
    user_turns = [turn for turn in user_turns if turn]
    return " ".join(user_turns[-3:]) or _latest_user_message(messages)


def answer_chat(
    messages: list[dict],
    workspace: str,
    top_k: int | None = None,
    model: str | None = None,
    filenames: list[str] | None = None,
) -> dict:
    """Repond a un tour de conversation en s'appuyant sur les documents du workspace.

    Le retrieval est pilote par le **dernier message utilisateur** ; le prompt inclut
    les derniers tours de la conversation (memoire multi-tours). Le serveur reste
    **stateless** : l'historique est fourni par l'appelant. La recherche peut etre
    restreinte a un sous-ensemble de documents (`filenames`).
    """
    k = top_k or settings.top_k
    model_name = model or settings.ollama_model

    # Question globale (« de quoi parle... », « resume ») : synthese sur tout le
    # document plutot que retrieval top-k. Le routage porte sur la DERNIERE question.
    latest = _latest_user_message(messages)
    if _route_is_summary(latest, model_name):
        return summarize(workspace, latest, model_name, filenames=filenames)

    # Retrieval pilote par les dernieres questions utilisateur (coreference sur les
    # follow-ups) ; la generation, elle, recoit tout l'historique recent.
    query = _retrieval_query(messages)
    passages, sources = _search_sources(query, workspace, k, filenames=filenames)

    prompt = build_chat_prompt(messages, passages)
    answer = get_llm(model_name).invoke(prompt)
    return {
        "answer": answer,
        "sources": sources,
        "model": model_name,
        "cited": cited_indices(answer, len(sources)),
    }


def _build_source(document, score) -> dict:
    """Construit une source d'affichage, enrichie de l'instant video si transcript."""
    metadata = document.metadata
    source = {
        "filename": metadata.get("filename", ""),
        "passage_id": metadata.get("passage_id", -1),
        "excerpt": metadata.get("excerpt", ""),
        # Score de reranking (plus grand = mieux) si reranking actif, sinon None.
        "score": float(score) if score is not None else None,
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
