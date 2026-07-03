"""Point d'entree de l'API FastAPI (Assistant documentaire Lite RAG)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Assistant documentaire Lite RAG",
    description="Chaine RAG legere : upload / index / ask, sur un document unique.",
    version="1.0.0",
)

# CORS ouvert : l'interface Streamlit consomme l'API depuis le navigateur.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
