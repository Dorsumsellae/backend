"""Extraction du texte brut a partir des documents uploades.

Formats pris en charge :
- Texte : .txt, .md, .markdown (decodage UTF-8 direct, sans dependance externe).
- PDF non scanne : .pdf (extraction de la couche texte via `pypdf`).

Les PDF **scannes** (images sans couche texte) ne sont pas geres : ils
necessiteraient de l'OCR, hors perimetre. On leve alors une erreur explicite.

`pypdf` est importe paresseusement : le chemin texte reste autonome (utile pour
la CI legere), seul le traitement d'un vrai PDF charge la dependance.
"""

import os
from io import BytesIO

# Extensions traitees comme texte brut.
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}
PDF_EXTENSIONS = {".pdf"}
# Transcripts horodates : traites a part par `app.rag.transcripts` a l'indexation,
# mais acceptes a l'upload au meme titre que les autres formats.
TRANSCRIPT_EXTENSIONS = {".srt", ".vtt"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | TRANSCRIPT_EXTENSIONS


def is_supported(filename: str) -> bool:
    """Indique si l'extension du fichier fait partie des formats geres."""
    return _extension(filename) in SUPPORTED_EXTENSIONS


def extract_text(filename: str, content: bytes) -> str:
    """Retourne le texte brut d'un document a partir de son nom et de ses octets.

    Le format est deduit de l'extension de `filename`.

    Raises:
        ValueError: extension non supportee, PDF sans texte extractible
            (probablement scanne), ou texte non decodable en UTF-8.
    """
    ext = _extension(filename)
    if ext in TEXT_EXTENSIONS:
        return _decode_text(content)
    if ext in PDF_EXTENSIONS:
        return _extract_pdf(content)
    raise ValueError(
        f"Format non pris en charge : '{ext or filename}'. "
        f"Formats acceptes : {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


def _extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Le fichier texte n'est pas encode en UTF-8."
        ) from exc


def _extract_pdf(content: bytes) -> str:
    """Extrait la couche texte d'un PDF non scanne (import paresseux de pypdf)."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise ValueError(f"PDF illisible ou corrompu : {exc}") from exc

    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text.strip():
        raise ValueError(
            "Aucun texte extractible du PDF : il est probablement scanne "
            "(image). L'OCR n'est pas pris en charge."
        )
    return text
