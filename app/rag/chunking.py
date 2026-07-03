"""Decoupage du texte en passages courts (chunks).

Deux strategies sont disponibles (cf. techniques_rag_contexte.md, section 1) :

- "fixed"     : Fixed-size. Coupe tous les N caracteres avec recouvrement.
                Simple, independant de LangChain (donc validable en CI).
                C'est la strategie **par defaut**.
- "recursive" : Recursive splitting. Coupe en respectant paragraphes -> phrases
                -> mots via le RecursiveCharacterTextSplitter de LangChain, pour
                eviter de couper au milieu d'une phrase.

La strategie par defaut est fixee par `settings.chunk_strategy` et peut etre
surchargee appel par appel. LangChain n'est importe que si "recursive" est
reellement demande (import paresseux), afin que "fixed" reste autonome.
"""

from dataclasses import dataclass

from app.config import settings


@dataclass
class Chunk:
    passage_id: int
    text: str


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    strategy: str | None = None,
) -> list[Chunk]:
    """Decoupe `text` en passages numerotes a partir de 0.

    Args:
        text: texte source.
        chunk_size: taille d'un passage (defaut : settings.chunk_size).
        chunk_overlap: recouvrement entre deux passages (defaut : settings.chunk_overlap).
        strategy: "fixed" ou "recursive" (defaut : settings.chunk_strategy).

    Returns:
        Liste de Chunk numerotes a partir de 0.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    strategy = (strategy or settings.chunk_strategy).lower()

    if size <= 0:
        raise ValueError("chunk_size doit etre strictement positif.")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap doit verifier : 0 <= overlap < chunk_size.")

    text = text.strip()
    if not text:
        return []

    if strategy == "fixed":
        segments = _split_fixed(text, size, overlap)
    elif strategy == "recursive":
        segments = _split_recursive(text, size, overlap)
    else:
        raise ValueError(
            f"Strategie de chunking inconnue : {strategy!r} "
            "(valeurs acceptees : 'fixed', 'recursive')."
        )

    return [Chunk(passage_id=i, text=segment) for i, segment in enumerate(segments)]


def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size : fenetre glissante de `size` caracteres, pas de `size - overlap`."""
    segments: list[str] = []
    step = size - overlap
    for start in range(0, len(text), step):
        segment = text[start : start + size].strip()
        if segment:
            segments.append(segment)
        if start + size >= len(text):
            break
    return segments


def _split_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Recursive splitting via LangChain (import paresseux : 'fixed' reste sans dep)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return [seg.strip() for seg in splitter.split_text(text) if seg.strip()]
