"""Decoupage du texte en passages courts (chunks).

Implementation simple et testable, independante de LangChain, afin de pouvoir
la valider en CI. Vous pouvez la remplacer par un
`RecursiveCharacterTextSplitter` de LangChain si vous le souhaitez.
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
) -> list[Chunk]:
    """Decoupe `text` en passages de `chunk_size` caracteres avec recouvrement.

    Args:
        text: texte source.
        chunk_size: taille d'un passage (defaut : settings.chunk_size).
        chunk_overlap: recouvrement entre deux passages consecutifs.

    Returns:
        Liste de Chunk numerotes a partir de 0.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if size <= 0:
        raise ValueError("chunk_size doit etre strictement positif.")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap doit verifier : 0 <= overlap < chunk_size.")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    passage_id = 0
    for start in range(0, len(text), step):
        segment = text[start : start + size].strip()
        if segment:
            chunks.append(Chunk(passage_id=passage_id, text=segment))
            passage_id += 1
        if start + size >= len(text):
            break
    return chunks
