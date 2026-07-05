"""Endpoints de l'API RAG.

    GET  /health          -> disponibilite du service
    POST /upload          -> envoi du document (stockage MinIO), range par workspace
    POST /index           -> indexation (texte, PDF, ou transcript horodate)
    POST /ingest/youtube  -> recupere les sous-titres d'une video YouTube et les indexe
    GET  /documents       -> liste des documents indexes d'un workspace
    GET  /workspaces      -> liste des workspaces existants
    GET  /models          -> liste des modeles Ollama disponibles
    POST /ask             -> question -> reponse generee + sources (cloisonnee au workspace)
    POST /chat            -> conversation multi-tours -> reponse + sources (cloisonnee)
    POST /reset           -> reinitialise l'indexation (workspace entier ou un document)
"""

import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from minio.error import S3Error

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentsResponse,
    IndexRequest,
    IndexResponse,
    IngestYoutubeRequest,
    IngestYoutubeResponse,
    MessageInfo,
    MessagesResponse,
    ModelInfo,
    ModelsResponse,
    NotebookCreate,
    NotebookInfo,
    NotebookRename,
    NotebooksResponse,
    NoteCreate,
    NoteInfo,
    NotesResponse,
    ResetRequest,
    ResetResponse,
    WorkspacesResponse,
)
from app.config import settings
from app.db import repository
from app.db.database import session_scope
from app.rag import loaders, pipeline, transcripts
from app.storage import minio_client
from app.workspaces import normalize_workspace

router = APIRouter()


def _resolve_workspace(value: str | None) -> str:
    """Valide un nom de workspace ou leve une 400 explicite."""
    try:
        return normalize_workspace(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _http_error_detail(exc: requests.RequestException) -> str:
    """Extrait le message d'erreur le plus parlant d'une exception `requests`.

    Quand le service distant repond un corps JSON `{"detail": "..."}` (cas FastAPI),
    on privilegie ce message ; sinon on retombe sur le texte brut de l'exception.
    Le frontend porte le meme utilitaire (`api_client.error_detail`) : les deux
    services etant deployes separement, ils ne peuvent pas partager ce code sans
    introduire un package commun.
    """
    detail = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
    return detail


@router.get("/health", tags=["health"])
def health() -> dict:
    """Endpoint de sante (utilise par la CI et le healthcheck Docker)."""
    return {"status": "ok"}


@router.post("/upload", tags=["rag"])
async def upload(
    file: UploadFile = File(...),
    workspace: str | None = Form(None),
) -> dict:
    """Recoit un document (texte ou PDF non scanne) et le stocke dans MinIO.

    Le document est range dans le `workspace` indique (defaut : workspace serveur).
    """
    ws = _resolve_workspace(workspace)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")
    if not loaders.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format non pris en charge. Formats acceptes : "
                f"{', '.join(sorted(loaders.SUPPORTED_EXTENSIONS))}."
            ),
        )
    content = await file.read()
    minio_client.put_document(
        minio_client.object_name(ws, file.filename),
        content,
        content_type=file.content_type,
    )
    return {"filename": file.filename, "workspace": ws}


@router.post("/index", response_model=IndexResponse, tags=["rag"])
def index(req: IndexRequest) -> IndexResponse:
    """Indexe un document deja stocke dans MinIO (texte extrait selon le format)."""
    ws = _resolve_workspace(req.workspace)
    try:
        content = minio_client.get_document_bytes(
            minio_client.object_name(ws, req.filename)
        )
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document '{req.filename}' introuvable dans le workspace "
                    f"'{ws}'. Uploadez-le d'abord via /upload."
                ),
            ) from exc
        raise

    # Un transcript horodate (.srt/.vtt, ou .txt '[HH:MM:SS] ...') suit un chemin
    # dedie qui preserve les instants ; les autres documents restent en texte brut.
    try:
        preview = content.decode("utf-8")
    except UnicodeDecodeError:
        preview = None

    try:
        if transcripts.is_transcript(req.filename, preview):
            cues = transcripts.parse_transcript(req.filename, content)
            chunks_indexed = pipeline.index_transcript(req.filename, cues, workspace=ws)
        else:
            text = loaders.extract_text(req.filename, content)
            chunks_indexed = pipeline.index_document(
                req.filename, text, workspace=ws, strategy=req.strategy
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndexResponse(
        filename=req.filename, workspace=ws, chunks_indexed=chunks_indexed
    )


def _detect_original_language(transcript_list) -> str | None:
    """Devine la langue originale (parlee) de la video parmi les pistes disponibles.

    La piste **auto-generee** par YouTube reflete la langue effectivement parlee
    (les autres langues en sont des traductions). A defaut, on retient la premiere
    piste disponible. Retourne None si la liste est vide.
    """
    for transcript in transcript_list:
        if transcript.is_generated:
            return transcript.language_code
    for transcript in transcript_list:
        return transcript.language_code
    return None


class _NoCaptionsAvailable(Exception):
    """La video n'expose aucune piste de sous-titres : candidat au fallback ASR."""


def _fetch_youtube_transcript(
    video_id: str, preferred_languages: list[str] | None
) -> tuple[list, str]:
    """Recupere les sous-titres d'une video (import paresseux de la dependance).

    Sans `preferred_languages`, on prend la **langue originale** de la video
    (piste auto-generee), avec un repli sur l'anglais.

    Returns:
        (cues, code_langue) : liste de `transcripts.Cue` (speaker None) + langue.

    Raises:
        _NoCaptionsAvailable: aucune piste de sous-titres (declenche le fallback ASR).
        HTTPException: dependance absente (500), video indisponible (422),
            ou echec reseau/YouTube (502).
    """
    try:
        from youtube_transcript_api import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Dependance 'youtube-transcript-api' absente cote serveur.",
        ) from exc

    try:
        # API 1.x : instance + list()/find_transcript(). On liste d'abord les pistes
        # pour determiner la langue originale, puis on recupere avec repli sur l'anglais.
        transcript_list = YouTubeTranscriptApi().list(video_id)
        if preferred_languages:
            languages = preferred_languages
        else:
            original = _detect_original_language(transcript_list)
            # Langue originale d'abord, repli sur l'anglais ('dict.fromkeys' dedoublonne
            # en conservant l'ordre, au cas ou l'original serait deja 'en').
            languages = list(dict.fromkeys(l for l in (original, "en") if l))
        transcript = transcript_list.find_transcript(languages)
        raw = transcript.fetch().to_raw_data()  # dicts {text, start, duration}
        cues = [
            transcripts.Cue(
                start=float(item["start"]),
                end=float(item["start"]) + float(item.get("duration", 0.0)),
                text=transcripts.clean_text(item["text"]),
            )
            for item in raw
        ]
        return cues, transcript.language_code
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        # Pas de piste de sous-titres : on laisse l'appelant tenter l'ASR.
        raise _NoCaptionsAvailable(str(exc)) from exc
    except VideoUnavailable as exc:
        raise HTTPException(
            status_code=422, detail=f"Video YouTube indisponible : {exc}"
        ) from exc
    except Exception as exc:  # reseau, blocage IP YouTube, format inattendu...
        raise HTTPException(
            status_code=502,
            detail=f"Echec de recuperation du transcript YouTube : {exc}",
        ) from exc


def _asr_transcribe(
    video_id: str,
    preferred_languages: list[str] | None,
    num_speakers: int | None,
) -> tuple[list, str, bool]:
    """Transcrit la video via le service ASR isole (fallback sans sous-titres).

    Returns:
        (cues, langue, diarized) : cues `transcripts.Cue` (avec speaker si diarise).
    """
    payload: dict = {"video_id": video_id, "diarize": True}
    if preferred_languages:
        payload["languages"] = preferred_languages
    if num_speakers:
        payload["num_speakers"] = num_speakers

    try:
        response = requests.post(
            f"{settings.asr_service_url.rstrip('/')}/transcribe",
            json=payload,
            timeout=settings.asr_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Transcription ASR echouee : {_http_error_detail(exc)}",
        ) from exc

    data = response.json()
    cues = [
        transcripts.Cue(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=transcripts.clean_text(seg["text"]),
            speaker=seg.get("speaker"),
        )
        for seg in data.get("segments", [])
    ]
    return cues, data.get("language", ""), bool(data.get("diarized"))


@router.post("/ingest/youtube", response_model=IngestYoutubeResponse, tags=["rag"])
def ingest_youtube(req: IngestYoutubeRequest) -> IngestYoutubeResponse:
    """Indexe une video YouTube comme transcript horodate.

    Chaine : on tente d'abord les **sous-titres** YouTube (langue originale + repli
    anglais) ; si la video n'en a pas, on bascule sur le **service ASR** (transcription
    audio + diarisation) lorsqu'il est configure. Le transcript est archive dans MinIO
    puis indexe avec l'URL video : les sources /ask pointent vers l'instant exact.
    """
    ws = _resolve_workspace(req.workspace)
    try:
        video_id = transcripts.extract_video_id(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    diarized = False
    try:
        cues, language = _fetch_youtube_transcript(video_id, req.languages)
        provenance = "captions"
    except _NoCaptionsAvailable as exc:
        if not settings.asr_service_url:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Cette video n'a pas de sous-titres. Importez un fichier .srt/.vtt "
                    "via /upload, ou activez le service ASR (ASR_SERVICE_URL)."
                ),
            ) from exc
        cues, language, diarized = _asr_transcribe(
            video_id, req.languages, req.num_speakers
        )
        provenance = "asr"

    cues = [cue for cue in cues if cue.text]
    if not cues:
        raise HTTPException(
            status_code=422, detail="Le transcript recupere est vide apres nettoyage."
        )

    source_url = f"https://www.youtube.com/watch?v={video_id}"
    filename = f"youtube_{video_id}.txt"

    # Archivage du transcript horodate (permet une reindexation ulterieure via /index).
    archive = transcripts.cues_to_bracketed_text(cues).encode("utf-8")
    minio_client.put_document(
        minio_client.object_name(ws, filename), archive, content_type="text/plain"
    )

    chunks_indexed = pipeline.index_transcript(
        filename, cues, workspace=ws, source_url=source_url
    )
    return IngestYoutubeResponse(
        filename=filename,
        workspace=ws,
        video_id=video_id,
        source_url=source_url,
        language=language,
        provenance=provenance,
        diarized=diarized,
        chunks_indexed=chunks_indexed,
    )


@router.get("/documents", response_model=DocumentsResponse, tags=["rag"])
def documents(workspace: str | None = None) -> DocumentsResponse:
    """Liste les documents indexes dans un workspace (defaut : workspace serveur)."""
    ws = _resolve_workspace(workspace)
    docs = pipeline.list_indexed_documents(ws)
    return DocumentsResponse(
        workspace=ws,
        documents=[DocumentInfo(**doc) for doc in docs],
        count=len(docs),
    )


@router.get("/workspaces", response_model=WorkspacesResponse, tags=["rag"])
def workspaces() -> WorkspacesResponse:
    """Liste les workspaces contenant au moins un document indexe."""
    return WorkspacesResponse(
        workspaces=pipeline.list_workspaces(),
        default=settings.default_workspace,
    )


@router.post("/reset", response_model=ResetResponse, tags=["rag"])
def reset(req: ResetRequest | None = None) -> ResetResponse:
    """Reinitialise l'indexation d'un workspace.

    Corps optionnel : sans `filename`, tous les documents du workspace sont
    desindexes ; avec `{"filename": "..."}` seul ce document l'est. Les autres
    workspaces ne sont jamais touches.
    """
    ws = _resolve_workspace(req.workspace if req else None)
    filename = req.filename if req else None
    result = pipeline.reset_index(ws, filename)
    return ResetResponse(**result)


@router.get("/models", response_model=ModelsResponse, tags=["rag"])
def models() -> ModelsResponse:
    """Liste les modeles Ollama disponibles (pour alimenter le selecteur du front)."""
    try:
        names = pipeline.list_available_models()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Serveur Ollama injoignable : {exc}",
        ) from exc
    return ModelsResponse(
        models=[
            ModelInfo(name=name, is_default=(name == settings.ollama_model))
            for name in names
        ],
        default=settings.ollama_model,
    )


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Repond a une question a partir des documents du workspace indique.

    La recherche est cloisonnee au workspace ; `filename` la restreint en plus a
    un seul document.
    """
    ws = _resolve_workspace(req.workspace)
    result = pipeline.answer_question(
        req.question,
        workspace=ws,
        top_k=req.top_k,
        model=req.model,
        filename=req.filename,
        filenames=req.filenames,
    )
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
        cited=result.get("cited"),
    )


@router.post("/chat", response_model=ChatResponse, tags=["rag"])
def chat(req: ChatRequest) -> ChatResponse:
    """Repond a un tour de conversation, en s'appuyant sur les documents du workspace.

    Chat **multi-tours** : le client envoie l'historique complet (`messages`) ; le
    retrieval est pilote par le dernier message utilisateur et la reponse tient
    compte des echanges precedents. La recherche est cloisonnee au workspace et peut
    etre restreinte a un sous-ensemble de documents (`filenames`). Le serveur reste
    stateless (aucun stockage de session).
    """
    ws = _resolve_workspace(req.workspace)
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Le dernier message doit etre une question de l'utilisateur.",
        )
    result = pipeline.answer_chat(
        [message.model_dump() for message in req.messages],
        workspace=ws,
        top_k=req.top_k,
        model=req.model,
        filenames=req.filenames,
    )
    _persist_chat_turn(ws, req.messages[-1].content, result)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
        cited=result.get("cited"),
    )


def _persist_chat_turn(workspace: str, question: str, result: dict) -> None:
    """Sauvegarde best-effort du tour (question + reponse) ; n'echoue jamais le chat.

    Si la base est injoignable, on ignore silencieusement : le RAG reste fonctionnel,
    seule la persistance de l'historique est perdue.
    """
    try:
        with session_scope() as session:
            repository.get_or_create_notebook(session, workspace)
            repository.add_message(session, workspace, "user", question)
            repository.add_message(
                session,
                workspace,
                "assistant",
                result["answer"],
                sources=result.get("sources"),
                cited=result.get("cited"),
                model=result.get("model"),
            )
    except Exception:
        pass


# --- Notebooks / conversations / notes (persistance Postgres) ----------------


def _notebook_info(notebook) -> NotebookInfo:
    return NotebookInfo(
        id=notebook.id, title=notebook.title, created_at=notebook.created_at
    )


@router.get("/notebooks", response_model=NotebooksResponse, tags=["notebooks"])
def list_notebooks() -> NotebooksResponse:
    """Liste les notebooks persistes, en rattachant les workspaces deja indexes.

    Backfill paresseux : tout workspace present dans ChromaDB mais absent de la base
    recoit une entree notebook (title = id). Si la base est injoignable, on renvoie
    une liste degradee derivee des workspaces (sans persistance).
    """
    try:
        workspace_ids = pipeline.list_workspaces()
    except Exception:
        workspace_ids = []
    try:
        with session_scope() as session:
            repository.backfill_notebooks(session, workspace_ids)
            notebooks = [_notebook_info(n) for n in repository.list_notebooks(session)]
    except Exception:
        notebooks = [NotebookInfo(id=ws, title=ws) for ws in sorted(workspace_ids)]
    return NotebooksResponse(notebooks=notebooks, default=settings.default_workspace)


@router.post("/notebooks", response_model=NotebookInfo, tags=["notebooks"])
def create_notebook(req: NotebookCreate) -> NotebookInfo:
    """Cree un notebook a partir d'un titre libre (id = slug sur et unique)."""
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Titre de notebook requis.")
    try:
        with session_scope() as session:
            return _notebook_info(repository.create_notebook(session, title))
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Base de donnees injoignable : {exc}"
        ) from exc


@router.patch("/notebooks/{notebook_id}", response_model=NotebookInfo, tags=["notebooks"])
def rename_notebook(notebook_id: str, req: NotebookRename) -> NotebookInfo:
    """Renomme un notebook (titre libre) ; l'id/workspace ne change pas."""
    ws = _resolve_workspace(notebook_id)
    try:
        with session_scope() as session:
            notebook = repository.rename_notebook(session, ws, req.title)
            if notebook is None:
                raise HTTPException(status_code=404, detail="Notebook introuvable.")
            return _notebook_info(notebook)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Base de donnees injoignable : {exc}"
        ) from exc


@router.delete("/notebooks/{notebook_id}", tags=["notebooks"])
def delete_notebook(notebook_id: str) -> dict:
    """Supprime un notebook : son index ChromaDB puis sa ligne (cascade messages/notes)."""
    ws = _resolve_workspace(notebook_id)
    pipeline.reset_index(ws)  # vide la base vectorielle du workspace
    try:
        with session_scope() as session:
            repository.delete_notebook(session, ws)
    except Exception:
        pass
    return {"deleted": ws}


@router.get(
    "/notebooks/{notebook_id}/messages",
    response_model=MessagesResponse,
    tags=["notebooks"],
)
def get_messages(notebook_id: str) -> MessagesResponse:
    """Historique de conversation persiste (pour restaurer le chat a l'ouverture)."""
    ws = _resolve_workspace(notebook_id)
    try:
        with session_scope() as session:
            messages = [
                MessageInfo(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    sources=m.sources,
                    cited=m.cited,
                    model=m.model,
                    created_at=m.created_at,
                )
                for m in repository.list_messages(session, ws)
            ]
    except Exception:
        messages = []
    return MessagesResponse(notebook_id=ws, messages=messages)


@router.delete("/notebooks/{notebook_id}/messages", tags=["notebooks"])
def clear_messages(notebook_id: str) -> dict:
    """Efface l'historique de conversation d'un notebook."""
    ws = _resolve_workspace(notebook_id)
    removed = 0
    try:
        with session_scope() as session:
            removed = repository.clear_messages(session, ws)
    except Exception:
        pass
    return {"notebook_id": ws, "messages_removed": removed}


@router.get(
    "/notebooks/{notebook_id}/notes", response_model=NotesResponse, tags=["notebooks"]
)
def get_notes(notebook_id: str) -> NotesResponse:
    """Notes (panneau Studio) persistees d'un notebook."""
    ws = _resolve_workspace(notebook_id)
    try:
        with session_scope() as session:
            notes = [
                NoteInfo(id=n.id, text=n.text, created_at=n.created_at)
                for n in repository.list_notes(session, ws)
            ]
    except Exception:
        notes = []
    return NotesResponse(notebook_id=ws, notes=notes)


@router.post(
    "/notebooks/{notebook_id}/notes", response_model=NoteInfo, tags=["notebooks"]
)
def create_note(notebook_id: str, req: NoteCreate) -> NoteInfo:
    """Ajoute une note a un notebook."""
    ws = _resolve_workspace(notebook_id)
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Note vide.")
    try:
        with session_scope() as session:
            repository.get_or_create_notebook(session, ws)
            note = repository.add_note(session, ws, text)
            return NoteInfo(id=note.id, text=note.text, created_at=note.created_at)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Base de donnees injoignable : {exc}"
        ) from exc


@router.delete("/notebooks/{notebook_id}/notes/{note_id}", tags=["notebooks"])
def delete_note(notebook_id: str, note_id: int) -> dict:
    """Supprime une note d'un notebook."""
    ws = _resolve_workspace(notebook_id)
    deleted = False
    try:
        with session_scope() as session:
            deleted = repository.delete_note(session, ws, note_id)
    except Exception:
        pass
    if not deleted:
        raise HTTPException(status_code=404, detail="Note introuvable.")
    return {"deleted": note_id}
