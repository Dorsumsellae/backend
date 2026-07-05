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
from app.rag.pipeline import _retrieval_query, _where

# --- Helper _where (fonction pure) ------------------------------------------


def test_where_single_clause_is_flat():
    assert _where("ws") == {"workspace": "ws"}


def test_where_with_filename_uses_and_operator():
    assert _where("ws", "a.txt") == {
        "$and": [{"workspace": "ws"}, {"filename": "a.txt"}]
    }


def test_where_with_filenames_uses_in_operator():
    assert _where("ws", filenames=["a.txt", "b.txt"]) == {
        "$and": [{"workspace": "ws"}, {"filename": {"$in": ["a.txt", "b.txt"]}}]
    }


def test_where_single_filename_in_list_is_equality():
    # Une liste d'un element retombe sur une egalite simple (pas de $in inutile).
    assert _where("ws", filenames=["a.txt"]) == {
        "$and": [{"workspace": "ws"}, {"filename": "a.txt"}]
    }


def test_where_empty_filenames_is_workspace_only():
    # Liste vide => aucune clause filename (jamais {"$in": []} qui ne matcherait rien).
    assert _where("ws", filenames=[]) == {"workspace": "ws"}


def test_retrieval_query_injects_prior_user_turns():
    # Un follow-up elliptique doit voir le sujet des questions precedentes reinjecte.
    messages = [
        {"role": "user", "content": "Quel est le budget du projet Zephyr ?"},
        {"role": "assistant", "content": "3 millions d'euros."},
        {"role": "user", "content": "Et qui l'a lance ?"},
    ]
    query = _retrieval_query(messages)
    assert "Zephyr" in query  # sujet du 1er tour present dans la requete du follow-up
    assert "lance" in query  # question courante aussi


def test_retrieval_query_single_turn_is_the_question():
    messages = [{"role": "user", "content": "Une seule question"}]
    assert _retrieval_query(messages) == "Une seule question"


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


def test_ask_filenames_restricts_to_subset(store):
    pipeline.index_document("a.txt", "aaaa " * 100, workspace="alpha")
    pipeline.index_document("b.txt", "bbbb " * 100, workspace="alpha")
    pipeline.index_document("c.txt", "cccc " * 100, workspace="alpha")

    result = pipeline.answer_question(
        "q", workspace="alpha", top_k=10, filenames=["a.txt", "b.txt"]
    )
    filenames = {source["filename"] for source in result["sources"]}
    assert "c.txt" not in filenames  # le sous-ensemble exclut c.txt
    assert filenames <= {"a.txt", "b.txt"}


def test_sources_carry_sequential_cite_index(store):
    pipeline.index_document("a.txt", "contenu alpha", workspace="alpha")
    pipeline.index_document("b.txt", "contenu beta", workspace="alpha")

    result = pipeline.answer_question("une question", workspace="alpha", top_k=4)
    cites = [source["cite"] for source in result["sources"]]
    assert cites == list(range(1, len(result["sources"]) + 1))


def test_answer_chat_uses_last_user_message(store):
    pipeline.index_document("a.txt", "le chat dort " * 50, workspace="alpha")
    messages = [
        {"role": "user", "content": "premiere question"},
        {"role": "assistant", "content": "premiere reponse"},
        {"role": "user", "content": "seconde question"},
    ]
    result = pipeline.answer_chat(messages, workspace="alpha", top_k=4)

    assert result["answer"] == "REPONSE STUB"
    assert {source["filename"] for source in result["sources"]} == {"a.txt"}
    assert result["model"]  # modele renseigne


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


def test_list_documents_reports_type_and_source_url(store):
    from app.rag.transcripts import Cue

    pipeline.index_document("notes.txt", "texte simple " * 20, workspace="alpha")
    pipeline.index_transcript(
        "youtube_abc.txt",
        [Cue(start=0.0, end=5.0, text="intro de la video")],
        workspace="alpha",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
    )

    docs = {d["filename"]: d for d in pipeline.list_indexed_documents("alpha")}
    assert docs["notes.txt"]["type"] == "text"
    assert docs["notes.txt"]["source_url"] is None
    assert docs["youtube_abc.txt"]["type"] == "youtube"
    assert (
        docs["youtube_abc.txt"]["source_url"]
        == "https://www.youtube.com/watch?v=abcdefghijk"
    )


def test_reset_workspace_leaves_others_intact(store):
    pipeline.index_document("a.txt", "x " * 100, workspace="alpha")
    pipeline.index_document("b.txt", "y " * 100, workspace="beta")

    report = pipeline.reset_index("alpha")

    assert report["scope"] == "workspace"
    assert report["workspace"] == "alpha"  # requis par ResetResponse (route /reset)
    assert report["documents_removed"] == 1
    assert pipeline.list_workspaces() == ["beta"]  # beta intact


def test_reset_single_document(store):
    pipeline.index_document("a.txt", "x " * 100, workspace="alpha")
    pipeline.index_document("b.txt", "y " * 100, workspace="alpha")

    report = pipeline.reset_index("alpha", filename="a.txt")

    assert report["scope"] == "document"
    assert report["workspace"] == "alpha"  # requis par ResetResponse (route /reset)
    assert report["documents_removed"] == 1
    remaining = [d["filename"] for d in pipeline.list_indexed_documents("alpha")]
    assert remaining == ["b.txt"]
