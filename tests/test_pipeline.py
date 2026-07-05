"""Tests d'integration du pipeline RAG : cloisonnement par workspace.

Utilise un ChromaDB ephemere (en memoire) et des embeddings deterministes
(hachage local, sans telechargement) : aucun serveur ni reseau requis. Le LLM
Ollama est remplace par un stub. On valide surtout l'**isolation** : une
recherche dans un workspace ne doit jamais remonter les documents d'un autre.
"""

import hashlib

import pytest

# La stack lourde (ChromaDB + LangChain) n'a pas de wheel sur toutes les plateformes
# (ex. chroma-hnswlib sous Windows/Python 3.13) : on skippe proprement le module si
# elle est absente. La CI (Linux/Python 3.11) l'installe et execute donc ces tests.
pytest.importorskip("chromadb")
pytest.importorskip("langchain_chroma")
pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_huggingface")

from langchain_core.embeddings import Embeddings

from app.rag import pipeline
from app.rag.pipeline import _where

# --- Helper _where (fonction pure) ------------------------------------------


def test_where_single_clause_is_flat():
    assert _where("ws") == {"workspace": "ws"}


def test_where_with_filename_uses_and_operator():
    assert _where("ws", "a.txt") == {
        "$and": [{"workspace": "ws"}, {"filename": "a.txt"}]
    }


# --- Fixture : store ChromaDB ephemere + LLM stub ---------------------------


class _HashEmbeddings(Embeddings):
    """Embeddings deterministes locaux (hachage), suffisants pour tester le filtrage."""

    def embed_documents(self, texts):
        return [self._vec(text) for text in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:16]]


class _StubLLM:
    def invoke(self, prompt):
        return "REPONSE STUB"


# Compteur pour donner une collection distincte a chaque test : `EphemeralClient`
# partage un systeme en memoire entre instances, donc une collection au nom fixe
# fuirait ses donnees d'un test a l'autre.
_collection_counter = {"n": 0}


@pytest.fixture
def store(monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    from langchain_chroma import Chroma

    _collection_counter["n"] += 1
    client = chromadb.EphemeralClient()
    vectorstore = Chroma(
        client=client,
        collection_name=f"test_rag_{_collection_counter['n']}",
        embedding_function=_HashEmbeddings(),
    )
    monkeypatch.setattr(pipeline, "get_vectorstore", lambda: vectorstore)
    monkeypatch.setattr(pipeline, "get_llm", lambda model=None: _StubLLM())
    return vectorstore


# --- Indexation --------------------------------------------------------------


def test_index_returns_chunk_count(store):
    assert pipeline.index_document("a.txt", "phrase " * 300, workspace="alpha") > 0


def test_reindex_same_document_is_idempotent(store):
    text = "phrase " * 300
    n1 = pipeline.index_document("a.txt", text, workspace="alpha")
    count1 = store._collection.count()
    n2 = pipeline.index_document("a.txt", text, workspace="alpha")
    count2 = store._collection.count()
    assert n1 == n2
    assert count1 == count2  # pas d'accumulation de doublons


def test_same_filename_across_workspaces_do_not_collide(store):
    pipeline.index_document("doc.txt", "contenu alpha " * 50, workspace="alpha")
    pipeline.index_document("doc.txt", "contenu beta " * 50, workspace="beta")

    docs_alpha = pipeline.list_indexed_documents("alpha")
    docs_beta = pipeline.list_indexed_documents("beta")
    assert [d["filename"] for d in docs_alpha] == ["doc.txt"]
    assert [d["filename"] for d in docs_beta] == ["doc.txt"]
    # Les deux coexistent : le total est bien la somme des deux.
    assert store._collection.count() == (
        docs_alpha[0]["chunks_indexed"] + docs_beta[0]["chunks_indexed"]
    )


# --- Recherche cloisonnee (le bug corrige) ----------------------------------


def test_ask_is_scoped_to_workspace(store):
    pipeline.index_document("alpha.txt", "le chat dort. " * 50, workspace="alpha")
    pipeline.index_document("beta.txt", "le chien court. " * 50, workspace="beta")

    result = pipeline.answer_question("une question", workspace="alpha", top_k=4)

    filenames = {source["filename"] for source in result["sources"]}
    assert filenames == {"alpha.txt"}  # jamais beta.txt


def test_ask_returns_sources_from_multiple_files(store):
    # Deux documents courts (un passage chacun) dans le meme workspace.
    pipeline.index_document("a.txt", "contenu du document alpha", workspace="alpha")
    pipeline.index_document("b.txt", "contenu du document beta", workspace="alpha")

    # top_k >= nombre total de passages -> les deux fichiers doivent remonter.
    result = pipeline.answer_question("une question", workspace="alpha", top_k=4)

    filenames = {source["filename"] for source in result["sources"]}
    assert filenames == {"a.txt", "b.txt"}  # recherche bien multi-fichiers


def test_ask_filename_restricts_to_single_document(store):
    pipeline.index_document("a.txt", "aaaa " * 100, workspace="alpha")
    pipeline.index_document("b.txt", "bbbb " * 100, workspace="alpha")

    result = pipeline.answer_question(
        "q", workspace="alpha", top_k=4, filename="a.txt"
    )
    assert {source["filename"] for source in result["sources"]} == {"a.txt"}


# --- Transcript : metadonnees horodatees et sources cliquables --------------


def test_index_transcript_exposes_timecodes_and_url(store):
    from app.rag.transcripts import Cue

    cues = [
        Cue(start=0.0, end=5.0, text="introduction de la video sur le cloud"),
        Cue(start=5.0, end=10.0, text="deuxieme partie sur le rag et les embeddings"),
    ]
    n = pipeline.index_transcript(
        "youtube_abc.txt",
        cues,
        workspace="alpha",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    assert n >= 1

    result = pipeline.answer_question("une question", workspace="alpha", top_k=4)
    source = result["sources"][0]
    assert source["start_seconds"] is not None
    assert source["source_url"] == "https://www.youtube.com/watch?v=abcdefghijk"
    # Lien ancre a l'instant : &t=<secondes>s (l'URL contient deja '?v=').
    assert "&t=" in source["timecode_url"] and source["timecode_url"].endswith("s")


# --- Inventaire et reinitialisation -----------------------------------------


def test_list_workspaces(store):
    pipeline.index_document("a.txt", "x " * 100, workspace="alpha")
    pipeline.index_document("b.txt", "y " * 100, workspace="beta")
    assert pipeline.list_workspaces() == ["alpha", "beta"]


def test_reset_workspace_leaves_others_intact(store):
    pipeline.index_document("a.txt", "x " * 100, workspace="alpha")
    pipeline.index_document("b.txt", "y " * 100, workspace="beta")

    report = pipeline.reset_index("alpha")

    assert report["scope"] == "workspace"
    assert report["documents_removed"] == 1
    assert pipeline.list_workspaces() == ["beta"]  # beta intact


def test_reset_single_document(store):
    pipeline.index_document("a.txt", "x " * 100, workspace="alpha")
    pipeline.index_document("b.txt", "y " * 100, workspace="alpha")

    report = pipeline.reset_index("alpha", filename="a.txt")

    assert report["scope"] == "document"
    assert report["documents_removed"] == 1
    remaining = [d["filename"] for d in pipeline.list_indexed_documents("alpha")]
    assert remaining == ["b.txt"]
