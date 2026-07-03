"""Configuration centralisee, lue depuis les variables d'environnement.

Les valeurs par defaut correspondent au parametrage conseille dans le sujet :
    chunk_size    : 700 a 900 caracteres
    chunk_overlap : 100 a 150 caracteres
    top_k         : 3 a 4 passages
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- MinIO ---
    minio_endpoint: str = "minio:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # --- ChromaDB ---
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "rag_documents"

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:0.5b"

    # --- Embeddings ---
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- Parametres RAG ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4


settings = Settings()
