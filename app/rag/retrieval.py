"""Retrieval avance : hybrid (dense + BM25 / RRF), MMR, reranking, reorder.

Pipeline : candidats larges (dense/MMR + BM25 fusionnes par RRF) -> reranking par
cross-encoder -> top_k -> reorder anti « lost-in-the-middle ». Chaque etape est
desactivable par configuration (cf. `app.config`) ; les dependances lourdes sont
importees paresseusement et degradent gracieusement en cas d'echec (retour au dense).
"""

import re
from functools import lru_cache

from app.config import settings

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenisation simple (minuscules, mots alphanumeriques) pour BM25."""
    return _TOKEN_RE.findall((text or "").lower())


@lru_cache(maxsize=1)
def get_reranker():
    """Charge le cross-encoder de reranking (couteux ; telecharge au 1er appel, cache)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def _chunk_key(document) -> str:
    """Cle stable d'un chunk (pour dedoublonner/fusionner entre listes de candidats)."""
    metadata = document.metadata or {}
    return f"{metadata.get('filename', '')}:{metadata.get('passage_id', '')}"


def _dense_candidates(store, query, n, where):
    """Candidats par similarite vectorielle, avec MMR (diversite) si active."""
    if settings.use_mmr:
        return store.max_marginal_relevance_search(
            query,
            k=n,
            fetch_k=max(n * 2, 40),
            lambda_mult=settings.mmr_lambda,
            filter=where,
        )
    results = store.similarity_search_with_score(query, k=n, filter=where)
    return [document for document, _ in results]


def _sparse_candidates(store, query, n, where):
    """Candidats par BM25 (mots-cles) sur les chunks du perimetre (workspace/docs).

    Recupere les chunks du perimetre depuis ChromaDB et applique BM25 en memoire.
    Convient a un corpus petit/moyen (cf. limite connue : scan complet du perimetre).
    """
    from langchain_core.documents import Document
    from rank_bm25 import BM25Okapi

    data = store._collection.get(where=where, include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        return []
    bm25 = BM25Okapi([_tokenize(text) for text in documents])
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:n]
    return [
        Document(page_content=documents[i], metadata=metadatas[i] or {})
        for i in ranked
    ]


def _rrf_fuse(dense, sparse, n):
    """Fusionne deux listes classees par Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    docs_by_key: dict = {}
    for ranked in (dense, sparse):
        for rank, document in enumerate(ranked):
            key = _chunk_key(document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            docs_by_key.setdefault(key, document)
    best = sorted(scores, key=lambda key: scores[key], reverse=True)[:n]
    return [docs_by_key[key] for key in best]


def _rerank(query, documents, k):
    """Reordonne les (question, passage) par cross-encoder ; garde les k meilleurs."""
    model = get_reranker()
    scores = model.predict([(query, doc.page_content) for doc in documents])
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:k]]


def _reorder_lost_in_middle(scored):
    """Place les meilleurs passages en debut ET fin (les LLM negligent le milieu).

    Entree classee du meilleur au moins bon -> sortie alternee [1, 3, 5, ..., 6, 4, 2].
    """
    left, right = [], []
    for i, item in enumerate(scored):
        (left if i % 2 == 0 else right).append(item)
    return left + right[::-1]


def retrieve(store, query, k, where):
    """Retourne les k passages les plus pertinents : liste de (Document, score|None).

    Selon la configuration : candidats larges (dense/MMR + BM25/RRF) -> reranking
    cross-encoder -> reorder. `score` est le score de reranking (plus grand = mieux)
    quand le reranking est actif, sinon None.
    """
    top_n = (
        settings.retrieval_top_n
        if (settings.use_reranker or settings.use_hybrid)
        else k
    )
    candidates = _dense_candidates(store, query, top_n, where)

    if settings.use_hybrid:
        try:
            sparse = _sparse_candidates(store, query, top_n, where)
        except Exception:
            sparse = []
        if sparse:
            candidates = _rrf_fuse(candidates, sparse, top_n)

    if settings.use_reranker and candidates:
        try:
            scored = _rerank(query, candidates, k)
        except Exception:
            # Reranker indisponible (modele non telecharge, RAM...) : on garde l'ordre courant.
            scored = [(document, None) for document in candidates[:k]]
    else:
        scored = [(document, None) for document in candidates[:k]]

    if settings.reorder_lost_in_middle:
        scored = _reorder_lost_in_middle(scored)
    return scored
