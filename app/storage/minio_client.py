"""Client MinIO : stockage et lecture du document brut.

Si vous choisissez de ne pas utiliser MinIO (stockage local a la place),
expliquez ce choix dans le README (comme demande dans le sujet).
"""

import mimetypes
from functools import lru_cache
from io import BytesIO

from minio import Minio

from app.config import settings


def object_name(workspace: str, filename: str) -> str:
    """Cle d'objet MinIO d'un document, prefixee par son workspace.

    Deux workspaces peuvent ainsi stocker un fichier de meme nom sans collision
    (`{workspace}/{filename}`), a l'image du cloisonnement cote base vectorielle.
    """
    return f"{workspace}/{filename}"


@lru_cache(maxsize=1)
def get_client() -> Minio:
    """Retourne le client MinIO (et cree le bucket si necessaire)."""
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=settings.minio_secure,
    )
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)
    return client


def put_document(filename: str, content: bytes, content_type: str | None = None) -> str:
    """Stocke un document (octets bruts) dans le bucket et retourne son nom d'objet.

    `content_type` est deduit de l'extension si non fourni (ex. application/pdf),
    ce qui permet de conserver des documents binaires comme les PDF.
    """
    client = get_client()
    if not content_type:
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    client.put_object(
        settings.minio_bucket,
        filename,
        BytesIO(content),
        length=len(content),
        content_type=content_type,
    )
    return filename


def get_document_bytes(filename: str) -> bytes:
    """Recupere le contenu binaire brut d'un document stocke."""
    client = get_client()
    response = client.get_object(settings.minio_bucket, filename)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_document(filename: str) -> str:
    """Recupere le contenu texte (UTF-8) d'un document stocke.

    Conserve pour compatibilite ; pour les documents binaires (PDF), utiliser
    `get_document_bytes` puis `app.rag.loaders.extract_text`.
    """
    return get_document_bytes(filename).decode("utf-8")
