import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter

def tokenize(text: str) -> List[str]:
    """Simple tokenizer for English word tokens and CJK characters."""
    if not text:
        return []
    # Normalize case
    text = text.lower()
    # Match CJK characters or contiguous word/number tokens
    tokens = re.findall(r"[\u4e00-\u9fa5]|\b\w+\b", text)
    return tokens

class SimpleBM25:
    """Lightweight pure-Python BM25 similarity implementation.

    This avoids extra third-party tokenizers and supports quick Top-K reranking.
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)
            
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            
            for word in frequencies:
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size if self.corpus_size > 0 else 0
        
        # Compute IDF (Inverse Document Frequency)
        for word, freq in nd.items():
            # BM25 IDF formula
            idf_score = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf_score

    def get_score(self, query_tokens: List[str], index: int) -> float:
        """Return the score for one document against the query tokens."""
        score = 0.0
        doc_len = self.doc_len[index]
        frequencies = self.doc_freqs[index]
        
        for token in query_tokens:
            if token not in frequencies:
                continue
            
            freq = frequencies[token]
            idf = self.idf.get(token, 0)
            
            if self.avgdl == 0:
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1
            else:
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                
            score += idf * (numerator / denominator)
            
        return score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Return scores for all documents."""
        scores = [self.get_score(query_tokens, i) for i in range(self.corpus_size)]
        return scores


def get_top_k_bm25(query: str, all_results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Rerank collected results with BM25 and return the most relevant Top-K items.
    
    Args:
        query: Search query or task outline.
        all_results: Search results in [{"title": "xxx", "body": "xxx", "url": "xxx"}, ...] format.
        k: Number of top-scoring documents to keep.
        
    Returns:
        Ranked Top-K result list.
    """
    if not all_results or k <= 0:
        return []
    
    if len(all_results) <= k:
        return all_results
        
    # Build a corpus from title and body to measure relevance
    corpus = []
    for res in all_results:
        text = str(res.get("title", "")) + " " + str(res.get("body", "") or res.get("content", ""))
        tokens = tokenize(text)
        corpus.append(tokens)
        
    bm25 = SimpleBM25(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    # Sort by score in descending order
    scored_results = list(zip(scores, all_results))
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    # Return the original dictionaries for the Top-K items
    top_k_results = [item[1] for item in scored_results[:k]]

    # Emit lightweight debug information
    import logging
    logger = logging.getLogger('bm25')
    logger.info(f"--- BM25 Top K Results for query: '{query[:50]}...' ---")
    for i, (score, res) in enumerate(scored_results[:k]):
        logger.info(f"Top {i+1} [Score: {score:.4f}] - {res.get('title', 'No Title')} ({res.get('url') or res.get('href')})")

    return top_k_results


async def get_top_k_hybrid(
    query: str,
    all_results: List[Dict[str, Any]],
    k: int,
    embeddings,
    bm25_weight: float = 0.4,
    embedding_weight: float = 0.6,
    summary_max_chars: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Rerank results using a weighted blend of BM25 and embedding cosine similarity.

    Args:
        query: Search query or task outline.
        all_results: Result list in [{"title": "...", "body": "...", "url/href": "..."}] format.
        k: Number of top-scoring documents to keep.
        embeddings: LangChain embeddings instance supporting aembed_documents / aembed_query.
        bm25_weight: BM25 score weight, default 0.4.
        embedding_weight: Embedding similarity weight, default 0.6.

    Returns:
        Ranked Top-K result list.
    """
    import logging
    import numpy as np

    logger = logging.getLogger('bm25')

    if not all_results or k <= 0:
        return []
    if len(all_results) <= k:
        return all_results

    # Build text corpus from title plus body/snippet
    texts = []
    for res in all_results:
        body = str(res.get("body", "") or res.get("content", ""))
        if summary_max_chars is not None:
            body = body[:summary_max_chars]
        text = str(res.get("title", "")) + " " + body
        texts.append(text)

    # --- BM25 scores ---
    corpus = [tokenize(t) for t in texts]
    bm25 = SimpleBM25(corpus)
    query_tokens = tokenize(query)
    bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=float)
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max  # Normalize to [0, 1]

    # --- Embedding cosine similarity ---
    try:
        doc_embeddings = await embeddings.aembed_documents(texts)
        query_embedding = await embeddings.aembed_query(query)

        doc_vecs = np.array(doc_embeddings, dtype=float)
        q_vec = np.array(query_embedding, dtype=float)

        # Cosine similarity
        doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q_vec)
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
        q_norm = q_norm if q_norm > 0 else 1e-10
        emb_scores = (doc_vecs / doc_norms) @ (q_vec / q_norm)
        # Map from [-1, 1] to [0, 1]
        emb_scores = (emb_scores + 1) / 2
    except Exception as e:
        logger.warning(f"Embedding computation failed; falling back to BM25 only: {e}")
        emb_scores = np.zeros(len(all_results), dtype=float)
        bm25_weight, embedding_weight = 1.0, 0.0

    # --- Weighted fusion ---
    final_scores = bm25_weight * bm25_scores + embedding_weight * emb_scores

    scored_results = sorted(zip(final_scores, all_results), key=lambda x: x[0], reverse=True)
    top_k_results = [item[1] for item in scored_results[:k]]

    # --- Detailed logging: all candidates plus Top-K selections ---
    trunc_label = f", summary_max_chars={summary_max_chars}" if summary_max_chars is not None else ""
    logger.info(f"--- Hybrid candidates (total={len(scored_results)}, BM25={bm25_weight}, Emb={embedding_weight}{trunc_label}) query='{query[:60]}' ---")
    for i, (score, res) in enumerate(scored_results):
        href = res.get("url") or res.get("href", "")
        is_poison = "[poison]" if "http://research/" in href else "[regular]"
        snippet = (str(res.get("body", "") or res.get("content", ""))[:60]).replace("\n", " ")
        logger.info(f"  [{i+1}] {is_poison} score={score:.4f} title={res.get('title', '')[:40]} snippet={snippet}")

    logger.info(f"--- Hybrid Top-K={k} selected results ---")
    for i, (score, res) in enumerate(scored_results[:k]):
        href = res.get("url") or res.get("href", "")
        is_poison = "[poison]" if "http://research/" in href else "[regular]"
        logger.info(f"  ✅ [{i+1}] {is_poison} score={score:.4f} - {res.get('title', '')[:40]} ({href[:60]})")

    if len(scored_results) > k:
        logger.info(f"--- Filtered out ({len(scored_results) - k}) ---")
        for i, (score, res) in enumerate(scored_results[k:]):
            href = res.get("url") or res.get("href", "")
            is_poison = "[poison]" if "http://research/" in href else "[regular]"
            logger.info(f"  ❌ {is_poison} score={score:.4f} - {res.get('title', '')[:40]} ({href[:60]})")

    return top_k_results
