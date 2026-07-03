"""Client MinIO : stockage et lecture du document brut.

Si vous choisissez de ne pas utiliser MinIO (stockage local a la place),
expliquez ce choix dans le README (comme demande dans le sujet).
"""

from functools import lru_cache
from io import BytesIO

from minio import Minio

from app.config import settings


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


def put_document(filename: str, content: bytes) -> str:
    """Stocke un document dans le bucket et retourne son nom d'objet."""
    client = get_client()
    client.put_object(
        settings.minio_bucket,
        filename,
        BytesIO(content),
        length=len(content),
        content_type="text/plain",
    )
    return filename


def get_document(filename: str) -> str:
    """Recupere le contenu texte d'un document stocke."""
    client = get_client()
    response = client.get_object(settings.minio_bucket, filename)
    try:
        return response.read().decode("utf-8")
    finally:
        response.close()
        response.release_conn()
