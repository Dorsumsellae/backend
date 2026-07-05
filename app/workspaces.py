"""Notion d'espace de travail (workspace) : cloisonnement logique des documents.

Un workspace regroupe un ensemble de documents indexes. Il n'existe qu'une seule
collection ChromaDB : l'isolation est **logique**, portee par une metadonnee
`workspace` posee sur chaque passage et filtree a la recherche (cf. `app.rag.pipeline`).

Le nom d'un workspace sert aussi de prefixe de cle d'objet MinIO
(`{workspace}/{filename}`) : il doit donc rester un identifiant simple et sur.
"""

import re

from app.config import settings

# Identifiant sur : lettres/chiffres ASCII, tiret, souligne, point ; 1 a 64 caracteres.
# Volontairement restrictif pour rester utilisable comme prefixe de cle MinIO et
# comme valeur de metadonnee ChromaDB, sans risque d'injection ni de collision.
_WORKSPACE_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def normalize_workspace(value: str | None) -> str:
    """Valide et normalise un nom de workspace, en repliant sur la valeur par defaut.

    Args:
        value: nom fourni par l'appelant (peut etre None ou vide -> defaut).

    Returns:
        Le nom de workspace valide (sans espaces superflus).

    Raises:
        ValueError: le nom ne respecte pas le format autorise.
    """
    workspace = (value or "").strip() or settings.default_workspace
    if not _WORKSPACE_RE.fullmatch(workspace):
        raise ValueError(
            f"Nom de workspace invalide : {value!r}. "
            "Autorises : lettres, chiffres, '.', '-', '_' (1 a 64 caracteres)."
        )
    return workspace
