"""Tests du repository (notebooks/messages/notes) sur SQLite en memoire."""

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import models  # noqa: F401 — enregistre les modeles sur Base
from app.db import repository as repo
from app.db.database import Base


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        yield db
    finally:
        db.close()


def test_create_notebook_slugifies_title(session):
    notebook = repo.create_notebook(session, "Warhammer paint")
    assert notebook.id == "warhammer-paint"
    assert notebook.title == "Warhammer paint"


def test_create_notebook_ids_are_unique(session):
    a = repo.create_notebook(session, "Mon projet")
    b = repo.create_notebook(session, "Mon projet")
    assert a.id == "mon-projet"
    assert b.id == "mon-projet-2"


def test_backfill_creates_missing_notebooks(session):
    repo.create_notebook(session, "Existant")  # id "existant"
    repo.backfill_notebooks(session, ["existant", "default", "alpha"])
    ids = {n.id for n in repo.list_notebooks(session)}
    assert {"existant", "default", "alpha"} <= ids


def test_messages_roundtrip_and_clear(session):
    repo.get_or_create_notebook(session, "nb1", "NB1")
    repo.add_message(session, "nb1", "user", "salut")
    repo.add_message(
        session, "nb1", "assistant", "bonjour [1]",
        sources=[{"filename": "a.txt"}], cited=[1], model="qwen2.5:0.5b",
    )
    messages = repo.list_messages(session, "nb1")
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].sources == [{"filename": "a.txt"}]
    assert messages[1].cited == [1]
    assert repo.clear_messages(session, "nb1") == 2
    assert repo.list_messages(session, "nb1") == []


def test_notes_crud(session):
    repo.get_or_create_notebook(session, "nb1", "NB1")
    note = repo.add_note(session, "nb1", "ma note")
    assert [n.text for n in repo.list_notes(session, "nb1")] == ["ma note"]
    assert repo.delete_note(session, "nb1", note.id) is True
    assert repo.list_notes(session, "nb1") == []


def test_delete_notebook_cascades_messages_and_notes(session):
    repo.get_or_create_notebook(session, "nb1", "NB1")
    repo.add_message(session, "nb1", "user", "x")
    repo.add_note(session, "nb1", "y")
    assert repo.delete_notebook(session, "nb1") is True
    assert repo.list_messages(session, "nb1") == []
    assert repo.list_notes(session, "nb1") == []
