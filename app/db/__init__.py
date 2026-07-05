"""Couche de persistance applicative (SQLAlchemy / Postgres).

Stocke les metadonnees NON vectorielles : notebooks (titre libre + slug), historique
de conversation et notes. Les documents/embeddings restent dans ChromaDB/MinIO.
"""
