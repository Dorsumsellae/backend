"""Point d'entree de l'API FastAPI (Assistant documentaire Lite RAG)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise la base applicative au demarrage (best-effort).

    Si Postgres est injoignable, l'API demarre quand meme : le RAG (upload/index/
    chat) reste operationnel, seule la persistance notebooks/chat/notes est perdue.
    """
    try:
        from app.db.database import init_db

        init_db()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Assistant documentaire Lite RAG",
    description="Chaine RAG legere : upload / index / ask / chat, avec notebooks persistes.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS ouvert : l'interface Streamlit consomme l'API depuis le navigateur.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
