"""Traitement des transcripts (sous-titres) horodates, notamment YouTube.

Un transcript brut est bruite pour du RAG : horodatages inseres dans le texte,
annotations non verbales (`[rires]`, `[musique]`), balises de sous-titrage
(`<c>`, `<00:00:00.000>`), lignes tres fragmentees. Ce module :

1. **parse** les formats courants en une liste de `Cue` (start, end, texte) :
   - `.srt` / `.vtt` (WebVTT) ;
   - `.txt` horodate `[HH:MM:SS] texte` (format d'export type YouTube) ;
2. **nettoie** le texte (retrait des annotations/balises, espaces normalises) ;
3. **decoupe** en passages en **respectant les frontieres temporelles** : chaque
   passage conserve son `start`/`end`, ce qui permet des sources cliquables
   pointant vers l'instant precis de la video.

Tout est en pur Python (aucune dependance lourde), donc validable en CI legere.
"""

import os
import re
from dataclasses import dataclass

# --- Nettoyage --------------------------------------------------------------

_VTT_TAG_RE = re.compile(r"<[^>]*>")  # balises WebVTT : <c>, <00:00:00.000>, ...
_BRACKET_RE = re.compile(r"\[[^\]]*\]")  # annotations : [rires], [musique], [applaudissements]


def clean_text(text: str) -> str:
    """Retire les balises/annotations et normalise les espaces d'un fragment."""
    text = _VTT_TAG_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- Modele -----------------------------------------------------------------


@dataclass
class Cue:
    """Un segment horodate : `text` prononce entre `start` et `end` (secondes).

    `speaker` (optionnel) identifie le locuteur (ex. "SPEAKER_00") quand la source
    est diarisee (transcription ASR) ; None pour des sous-titres classiques.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptChunk:
    """Passage indexable regroupant plusieurs cues consecutives d'un meme locuteur."""

    passage_id: int
    text: str
    start: float
    end: float
    speaker: str | None = None


# --- Detection --------------------------------------------------------------

TRANSCRIPT_EXTENSIONS = {".srt", ".vtt"}

# Horodatage de debut de ligne au format [H:MM:SS] ou [MM:SS] (export YouTube).
_BRACKET_TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]")


def is_transcript(filename: str, text: str | None) -> bool:
    """Indique si le document doit etre traite comme un transcript horodate.

    - `.srt` / `.vtt` : toujours un transcript.
    - `.txt` : transcript si le contenu comporte plusieurs horodatages `[..:..]`.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in TRANSCRIPT_EXTENSIONS:
        return True
    if text is None:
        return False
    # Au moins 3 horodatages : evite les faux positifs sur un texte contenant
    # incidemment un `[12:34]`.
    return len(_BRACKET_TS_RE.findall(text)) >= 3


# --- Parsing : format horodate entre crochets ([HH:MM:SS] texte) ------------


def _bracket_seconds(match: re.Match) -> float:
    a, b, c = match.group(1), match.group(2), match.group(3)
    if c is None:
        return int(a) * 60 + int(b)  # [MM:SS]
    return int(a) * 3600 + int(b) * 60 + int(c)  # [HH:MM:SS]


def parse_bracketed(text: str) -> list[Cue]:
    """Parse un transcript `[HH:MM:SS] texte` en cues.

    Le texte d'une cue est celui compris entre son horodatage et le suivant ;
    la fin d'une cue est le debut de la suivante (la derniere reste ouverte).
    """
    matches = list(_BRACKET_TS_RE.finditer(text))
    cues: list[Cue] = []
    for i, match in enumerate(matches):
        start = _bracket_seconds(match)
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = clean_text(text[match.end() : end_pos])
        end = _bracket_seconds(matches[i + 1]) if i + 1 < len(matches) else start
        if segment:
            cues.append(Cue(start=start, end=end, text=segment))
    return cues


# --- Parsing : SRT / WebVTT --------------------------------------------------

# Ligne de temps SRT/VTT : "[HH:]MM:SS,mmm --> [HH:]MM:SS,mmm" (virgule ou point).
_CUE_TIME_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*"
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)


def _cue_seconds(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_cue_format(text: str) -> list[Cue]:
    """Parse un contenu SRT ou WebVTT (blocs separes par des lignes vides)."""
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        match = _CUE_TIME_RE.search(lines[time_index])
        if not match:
            continue
        start = _cue_seconds(*match.group(1, 2, 3, 4))
        end = _cue_seconds(*match.group(5, 6, 7, 8))
        body = clean_text(" ".join(lines[time_index + 1 :]))
        if body:
            cues.append(Cue(start=start, end=end, text=body))
    return cues


def parse_transcript(filename: str, content: bytes) -> list[Cue]:
    """Parse le contenu d'un transcript selon son extension. Leve si vide.

    Raises:
        ValueError: contenu non decodable en UTF-8 ou sans aucun horodatage.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Le transcript n'est pas encode en UTF-8.") from exc

    ext = os.path.splitext(filename)[1].lower()
    cues = parse_cue_format(text) if ext in TRANSCRIPT_EXTENSIONS else parse_bracketed(text)
    if not cues:
        raise ValueError(
            "Aucun segment horodate detecte : le fichier n'est pas un transcript "
            "reconnu (.srt, .vtt, ou texte '[HH:MM:SS] ...')."
        )
    return cues


# --- Decoupage temporel ------------------------------------------------------


def _overlap_tail(cues: list[Cue], overlap: int) -> list[Cue]:
    """Cues de fin a reprendre dans le passage suivant (recouvrement ~`overlap` car.)."""
    if overlap <= 0:
        return []
    tail: list[Cue] = []
    total = 0
    for cue in reversed(cues):
        if tail and total + len(cue.text) > overlap:
            break
        tail.insert(0, cue)
        total += len(cue.text) + 1
    return tail


def chunk_cues(
    cues: list[Cue], chunk_size: int, chunk_overlap: int
) -> list[TranscriptChunk]:
    """Regroupe des cues en passages d'au plus `chunk_size` caracteres.

    Le decoupage respecte les frontieres de cues (jamais au milieu d'un segment) ;
    chaque passage porte le `start` de sa premiere cue et le `end` de sa derniere.
    Un recouvrement d'environ `chunk_overlap` caracteres est repris d'un passage a
    l'autre pour ne pas perdre le contexte a la jointure. Un **changement de
    locuteur** force une coupure (un passage = un seul locuteur), afin de garder
    l'attribution exploitable pour la recherche et les citations.
    """
    chunks: list[TranscriptChunk] = []
    current: list[Cue] = []

    def flush() -> None:
        if not current:
            return
        chunks.append(
            TranscriptChunk(
                passage_id=len(chunks),
                text=" ".join(cue.text for cue in current).strip(),
                start=current[0].start,
                end=current[-1].end,
                speaker=current[0].speaker,
            )
        )

    current_len = 0
    for cue in cues:
        text = cue.text.strip()
        if not text:
            continue
        speaker_change = bool(current) and cue.speaker != current[-1].speaker
        projected = current_len + (1 if current else 0) + len(text)
        if current and (projected > chunk_size or speaker_change):
            flush()
            # Recouvrement uniquement entre passages d'un meme locuteur.
            current = [] if speaker_change else _overlap_tail(current, chunk_overlap)
            current_len = sum(len(c.text) for c in current) + max(len(current) - 1, 0)
        current.append(Cue(start=cue.start, end=cue.end, text=text, speaker=cue.speaker))
        current_len += (1 if len(current) > 1 else 0) + len(text)
    flush()
    return chunks


def cues_to_bracketed_text(cues: list[Cue]) -> str:
    """Rend des cues au format `[HH:MM:SS] (locuteur) texte` (archivage/reindexation)."""
    lines = []
    for cue in cues:
        prefix = f"{cue.speaker}: " if cue.speaker else ""
        lines.append(f"[{format_timestamp(cue.start)}] {prefix}{cue.text}")
    return "\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """Formate des secondes en `HH:MM:SS`."""
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


# --- Identifiant de video YouTube -------------------------------------------

_VIDEO_ID_RE = re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/)([A-Za-z0-9_-]{11})")
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """Extrait l'identifiant a 11 caracteres d'une URL (ou identifiant) YouTube.

    Reconnait watch?v=, youtu.be/, /shorts/, /embed/ et un identifiant nu.

    Raises:
        ValueError: aucun identifiant reconnaissable.
    """
    candidate = url.strip()
    if _BARE_ID_RE.match(candidate):
        return candidate
    match = _VIDEO_ID_RE.search(candidate)
    if not match:
        raise ValueError(f"URL YouTube non reconnue : {url!r}.")
    return match.group(1)
