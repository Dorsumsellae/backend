"""Tests des briques de retrieval avance (fonctions pures + rerank stubbe)."""

import pytest

pytest.importorskip("pydantic_settings")  # app.config en depend

from app.rag import retrieval


class _Doc:
    """Document minimal (metadata + page_content) pour les tests."""

    def __init__(self, filename, passage_id, text=""):
        self.metadata = {"filename": filename, "passage_id": passage_id}
        self.page_content = text


def test_tokenize_lowercases_and_splits():
    assert retrieval._tokenize("Le Projet Zephyr, 2021 !") == [
        "le",
        "projet",
        "zephyr",
        "2021",
    ]


def test_reorder_lost_in_middle_puts_best_at_edges():
    scored = [("a", 5), ("b", 4), ("c", 3), ("d", 2), ("e", 1)]
    order = [item for item, _ in retrieval._reorder_lost_in_middle(scored)]
    assert order[0] == "a"  # meilleur en tete
    assert order[-1] == "b"  # 2e meilleur en queue
    assert set(order) == {"a", "b", "c", "d", "e"}


def test_rrf_fuse_prefers_items_ranked_high_in_both():
    a, b, c = _Doc("f", 1), _Doc("f", 2), _Doc("f", 3)
    dense = [a, b, c]  # a le mieux classe
    sparse = [c, a, b]  # a bien classe aussi
    fused = retrieval._rrf_fuse(dense, sparse, n=3)
    keys = [retrieval._chunk_key(d) for d in fused]
    assert keys[0] == "f:1"  # a (bien classe dans les deux) ressort en tete


def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    d1, d2, d3 = _Doc("f", 1, "x"), _Doc("f", 2, "y"), _Doc("f", 3, "z")

    class _StubCrossEncoder:
        def predict(self, pairs):
            return [0.1, 0.9, 0.5]  # le 2e passage est juge le plus pertinent

    monkeypatch.setattr(retrieval, "get_reranker", lambda: _StubCrossEncoder())
    ranked = retrieval._rerank("q", [d1, d2, d3], k=2)
    assert [retrieval._chunk_key(d) for d, _ in ranked] == ["f:2", "f:3"]
    assert ranked[0][1] == pytest.approx(0.9)
