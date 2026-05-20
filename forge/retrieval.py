"""Retrieval and reranking helpers used by the web-poison experiments.

URL candidates (real web results + virtual http://research/... poison URLs) are
jointly reranked using the linear blend from paper Eq. (1):

    score(d, t_i) = α · BM25(d, t_i) + (1 - α) · cos(φ(d), φ(t_i))

where φ is text-embedding-3-small and the release default is α = 0.4.
Only the top-k candidates (k = number of real URLs) proceed to synthesis, so
adversarial documents must displace genuine ones rather than merely expand the
evidence set.

``get_top_k_hybrid`` implements this blend; ``get_top_k_bm25`` provides the
BM25-only variant used in ablations.  ``filter_url_items_by_title_similarity``
and ``get_local_document_url_items`` handle the virtual-URL conversion and
pre-filtering steps that precede the joint reranking.
"""

from gpt_researcher.actions.planning_sources import (
    filter_url_items_by_title_similarity,
    get_local_document_url_items,
)
from gpt_researcher.utils.bm25 import SimpleBM25, get_top_k_bm25, get_top_k_hybrid, tokenize

__all__ = [
    "SimpleBM25",
    "filter_url_items_by_title_similarity",
    "get_local_document_url_items",
    "get_top_k_bm25",
    "get_top_k_hybrid",
    "tokenize",
]
