"""Tests du decoupage en chunks."""

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
