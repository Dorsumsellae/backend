"""Notebooks : derivation d'un identifiant sur (slug) depuis un titre libre.

Un « notebook » cote UI = un workspace cote RAG. Le titre est libre (ex.
"Warhammer paint") ; l'identifiant technique en est un slug sur, directement
utilisable comme nom de workspace (cf. `app.workspaces.normalize_workspace`) :
minuscules ASCII, chiffres, '.', '-', '_', borne a 64 caracteres.
"""

import re
import unicodedata

_INVALID_RE = re.compile(r"[^a-z0-9._-]+")
_DASHES_RE = re.compile(r"-{2,}")


def slugify(title: str) -> str:
    """Transforme un titre libre en identifiant sur. Retombe sur 'notebook' si vide."""
    normalized = (
        unicodedata.normalize("NFKD", title or "")
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = _INVALID_RE.sub("-", normalized.lower())
    slug = _DASHES_RE.sub("-", slug).strip("-._")
    slug = slug[:64].strip("-._")
    return slug or "notebook"
