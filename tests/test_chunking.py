"""Tests du decoupage en chunks."""

import pytest

from app.rag.chunking import split_text


def test_split_returns_chunks():
    text = "abcdefghij" * 20  # 200 caracteres
    chunks = split_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


def test_passage_ids_are_sequential():
    chunks = split_text("x" * 300, chunk_size=100, chunk_overlap=20)
    ids = [c.passage_id for c in chunks]
    assert ids == list(range(len(chunks)))


def test_overlap_is_applied():
    text = "".join(str(i % 10) for i in range(120))
    chunks = split_text(text, chunk_size=50, chunk_overlap=10)
    # La fin du 1er passage doit reapparaitre au debut du 2e (recouvrement).
    assert chunks[0].text[-10:] == chunks[1].text[:10]


def test_empty_text_returns_no_chunk():
    assert split_text("   ") == []


def test_default_strategy_is_fixed_size():
    # Le decoupage par defaut (fixed) borne la taille des passages et
    # ne depend pas de LangChain.
    chunks = split_text("z" * 300, chunk_size=100, chunk_overlap=20)
    assert all(len(c.text) <= 100 for c in chunks)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        split_text("bonjour le monde", strategy="semantic")


def test_recursive_strategy_splits_on_paragraphs():
    pytest.importorskip("langchain_text_splitters")
    text = "Premier paragraphe.\n\n" + "Phrase repetee. " * 40
    chunks = split_text(text, chunk_size=120, chunk_overlap=20, strategy="recursive")
    assert len(chunks) > 1
    assert [c.passage_id for c in chunks] == list(range(len(chunks)))
    assert all(c.text.strip() for c in chunks)
