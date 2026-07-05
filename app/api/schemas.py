"""Schemas Pydantic pour les requetes et reponses de l'API."""

from typing import Literal

from pydantic import BaseModel, Field

# Description reutilisee : le workspace cloisonne logiquement les documents.
_WORKSPACE_DESC = (
    "Espace de travail cible. Si omis (ou null), le workspace par defaut du "
    "serveur est utilise. Les documents et les recherches sont cloisonnes par workspace."
)


class IndexRequest(BaseModel):
    filename: str = Field(..., description="Nom du fichier a indexer (dans MinIO).")
    workspace: str | None = Field(None, description=_WORKSPACE_DESC)
    strategy: Literal["fixed", "recursive"] | None = Field(
        None,
        description=(
            "Strategie de decoupage : 'fixed' (taille fixe avec recouvrement) ou "
            "'recursive' (respecte paragraphes -> phrases -> mots). Si omis, la "
            "valeur par defaut du serveur est utilisee (chunk_strategy = 'fixed')."
        ),
    )


class IndexResponse(BaseModel):
    filename: str
    workspace: str = Field(..., description="Workspace dans lequel le document a ete indexe.")
    chunks_indexed: int


class ResetRequest(BaseModel):
    workspace: str | None = Field(None, description=_WORKSPACE_DESC)
    filename: str | None = Field(
        None,
        description=(
            "Document a desindexer dans le workspace. Si omis (ou null), tous les "
            "documents du workspace sont desindexes."
        ),
    )


class ResetResponse(BaseModel):
    scope: str = Field(
        ...,
        description="'workspace' (tout le workspace) ou 'document' (un seul fichier).",
    )
    workspace: str = Field(..., description="Workspace concerne par la reinitialisation.")
    documents_removed: int = Field(
        ..., description="Nombre de documents distincts desindexes."
    )
    chunks_removed: int = Field(..., description="Nombre de passages supprimes.")


class DocumentInfo(BaseModel):
    filename: str = Field(..., description="Nom du document indexe.")
    chunks_indexed: int = Field(
        ..., description="Nombre de passages (chunks) indexes pour ce document."
    )
    type: str = Field(
        "text",
        description="Origine/type de la source : 'text', 'pdf', 'youtube' ou 'transcript'.",
    )
    source_url: str | None = Field(
        None, description="URL de la source (ex. video YouTube), si connue."
    )


class DocumentsResponse(BaseModel):
    workspace: str = Field(..., description="Workspace interroge.")
    documents: list[DocumentInfo]
    count: int = Field(..., description="Nombre de documents indexes distincts.")


class WorkspacesResponse(BaseModel):
    workspaces: list[str] = Field(
        ..., description="Workspaces contenant au moins un document indexe."
    )
    default: str = Field(..., description="Workspace utilise quand aucun n'est precise.")


class AskRequest(BaseModel):
    question: str = Field(..., description="Question en langage naturel.")
    workspace: str | None = Field(None, description=_WORKSPACE_DESC)
    filename: str | None = Field(
        None,
        description=(
            "Restreint la recherche a ce seul document du workspace. Si omis, la "
            "recherche porte sur tous les documents du workspace."
        ),
    )
    filenames: list[str] | None = Field(
        None,
        description=(
            "Restreint la recherche a cet ensemble de documents du workspace. "
            "Ignore si `filename` (document unique) est fourni. Si omis ou vide, "
            "la recherche porte sur tous les documents du workspace."
        ),
    )
    top_k: int | None = Field(None, description="Nombre de passages a recuperer.")
    model: str | None = Field(
        None,
        description=(
            "Modele Ollama a utiliser pour la reponse (ex. 'qwen2.5:0.5b', "
            "'llama3.2:1b'). Si omis, le modele par defaut du serveur est utilise."
        ),
    )


class Source(BaseModel):
    filename: str
    passage_id: int
    excerpt: str
    score: float | None = None
    cite: int | None = Field(
        None,
        description=(
            "Index de citation du passage (1-based) : correspond aux marqueurs "
            "`[n]` presents dans la reponse, pour lier reponse et sources."
        ),
    )
    # Champs presents uniquement pour les passages issus d'un transcript horodate.
    start_seconds: float | None = Field(
        None, description="Instant de debut du passage dans la video (secondes)."
    )
    source_url: str | None = Field(
        None, description="URL de la source video (ex. YouTube), si connue."
    )
    timecode_url: str | None = Field(
        None, description="URL de la source ancree a l'instant du passage (`t=…s`)."
    )
    speaker: str | None = Field(
        None, description="Locuteur du passage (ex. 'SPEAKER_00'), si transcript diarise."
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str = Field(..., description="Modele Ollama ayant genere la reponse.")
    cited: list[int] | None = Field(
        None,
        description="Numeros de passages `[n]` reellement cites dans la reponse.",
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ..., description="Auteur du message : 'user' (question) ou 'assistant' (reponse)."
    )
    content: str = Field(..., description="Contenu textuel du message.")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        description=(
            "Historique complet de la conversation (le dernier message doit etre "
            "une question de role 'user'). Le serveur reste stateless : l'historique "
            "est fourni a chaque appel."
        ),
    )
    workspace: str | None = Field(None, description=_WORKSPACE_DESC)
    filenames: list[str] | None = Field(
        None,
        description=(
            "Restreint la recherche a cet ensemble de documents du workspace. Si "
            "omis ou vide, la recherche porte sur tous les documents du workspace."
        ),
    )
    top_k: int | None = Field(None, description="Nombre de passages a recuperer.")
    model: str | None = Field(
        None, description="Modele Ollama a utiliser (defaut : modele serveur)."
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str = Field(..., description="Modele Ollama ayant genere la reponse.")
    cited: list[int] | None = Field(
        None,
        description="Numeros de passages `[n]` reellement cites dans la reponse.",
    )


class IngestYoutubeRequest(BaseModel):
    url: str = Field(
        ...,
        description=(
            "URL ou identifiant d'une video YouTube (watch?v=, youtu.be/, "
            "/shorts/, /embed/, ou identifiant a 11 caracteres)."
        ),
    )
    workspace: str | None = Field(None, description=_WORKSPACE_DESC)
    languages: list[str] | None = Field(
        None,
        description=(
            "Langues de sous-titres preferees, par ordre de priorite (ex. ['fr', 'en']). "
            "Si omis : la langue originale de la video est utilisee, avec repli sur l'anglais."
        ),
    )
    num_speakers: int | None = Field(
        None,
        ge=1,
        description=(
            "Nombre de locuteurs (fallback ASR uniquement). 1 = mono-locuteur : "
            "diarisation ignoree (plus rapide). >1 = contrainte pour pyannote. "
            "None = detection automatique."
        ),
    )


class IngestYoutubeResponse(BaseModel):
    filename: str = Field(..., description="Nom logique du transcript indexe.")
    workspace: str
    video_id: str = Field(..., description="Identifiant de la video YouTube.")
    source_url: str = Field(..., description="URL canonique de la video.")
    language: str = Field(..., description="Langue des sous-titres / de la transcription.")
    provenance: str = Field(
        ..., description="'captions' (sous-titres YouTube) ou 'asr' (transcription audio)."
    )
    diarized: bool = Field(
        False, description="Vrai si des locuteurs ont ete attribues (transcription ASR)."
    )
    chunks_indexed: int


class ModelInfo(BaseModel):
    name: str = Field(..., description="Identifiant du modele Ollama.")
    is_default: bool = Field(
        ..., description="Vrai si c'est le modele par defaut du serveur."
    )


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default: str = Field(..., description="Modele utilise quand aucun n'est precise.")
